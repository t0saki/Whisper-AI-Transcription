import srt
from openai import OpenAI
import argparse
import os
from tqdm import tqdm
import yaml
import math

# Custom exceptions


class APIError(Exception):
    pass


class YAMLFormatError(Exception):
    pass

# Load configuration


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


API_KEY = load_config()["api_key"]
BASE_URL = load_config()["base_url"]
MODEL = load_config()["model"]

# Function to extract partial translations from a failed response


def extract_partial_translations(response_text: str) -> list[str]:
    """
    Extract as many valid translations as possible from a potentially invalid YAML response.

    Args:
        response_text: The raw response text from the API.

    Returns:
        A list of extracted translations, with empty strings for unparseable entries.
    """
    translations = []
    lines = response_text.splitlines()
    block = []
    for line in lines:
        if line.strip().startswith('- '):
            if block:
                # Process the previous block
                block_str = '\n'.join(block)
                try:
                    data = yaml.safe_load(block_str)
                    if isinstance(data, list) and len(data) == 1 and 'translation_text' in data[0]:
                        translations.append(data[0]['translation_text'])
                    else:
                        translations.append('')
                except:
                    translations.append('')
                block = [line]
            else:
                block.append(line)
        else:
            if block:
                block.append(line)
    # Process the last block
    if block:
        block_str = '\n'.join(block)
        try:
            data = yaml.safe_load(block_str)
            if isinstance(data, list) and len(data) == 1 and 'translation_text' in data[0]:
                translations.append(data[0]['translation_text'])
            else:
                translations.append('')
        except:
            translations.append('')
    return translations


def translate_batch(yaml_input: str, source_name: str = None, target_name: str = 'English') -> tuple[bool, str | None]:
    """
    Attempt to translate a batch of subtitles.

    Args:
        yaml_input: YAML formatted input string.
        source_name: Source language name.
        target_name: Target language name.

    Returns:
        Tuple (success, translated_yaml), where success is True if fully successful,
        and translated_yaml is the response or None if the API call fails.
    """
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    prompt = (
        f"You will be given a YAML formatted subtitles containing entries with 'id' and 'text' fields. Here is the input:\n\n"
        f"<yaml>\n{yaml_input}\n</yaml>\n\n"
        f"For each entry in the YAML, translate the contents of the 'text' field into {target_name}. "
        f"Write the translation back into the 'translation_text' field for that entry.\n\n"
        f"Here is an example of the expected format:\n\n"
        f"<example>\n"
        f"Input:\n"
        f"  - id: 1\n"
        f"    text: Hello\n"
        f"  - id: 2\n"
        f"    text: World\n"
        f"Output:\n"
        f"  - id: 1\n"
        f"    text: Hello\n"
        f"    translation_text: Hola\n"
        f"  - id: 2\n"
        f"    text: World\n"
        f"    translation_text: Mundo\n"
        f"</example>\n\n"
        f"Please return the translated YAML directly without wrapping <yaml> tag or including any additional information."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        translated_yaml = completion.choices[0].message.content.strip()
        try:
            translated_data = yaml.safe_load(translated_yaml)
            if not isinstance(translated_data, list):
                return False, translated_yaml
            for entry in translated_data:
                if 'id' not in entry or 'text' not in entry or 'translation_text' not in entry or not entry['translation_text'].strip():
                    return False, translated_yaml
            return True, translated_yaml
        except yaml.YAMLError:
            return False, translated_yaml
    except Exception as e:
        return False, None


def translate_srt(args):
    # Read SRT file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            subs = list(srt.parse(f))
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return
    except Exception as e:
        print(f"Error parsing SRT file: {e}")
        return

    # Language mapping
    lang_map = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
        "ko": "Korean", "zh": "Chinese"
    }
    source_name = lang_map.get(
        args.source.lower(), args.source) if args.source else None
    target_name = lang_map.get(args.target.lower(), args.target)

    # Dynamic batch sizing (aim for ~10 batches)
    total_subtitles = len(subs)
    optimal_batch_size = min(args.batch_size, max(
        1, math.ceil(total_subtitles / 10)))

    translated_texts = []
    max_retries = 4
    consecutive_failures = 0

    # Process batches
    for i in tqdm(range(0, total_subtitles, optimal_batch_size), desc="Translating subtitles"):
        batch_subs = subs[i:i + optimal_batch_size]
        batch_data = [{"id": j + 1, "text": sub.content}
                      for j, sub in enumerate(batch_subs)]
        yaml_input = yaml.dump(batch_data, allow_unicode=True)

        for attempt in range(max_retries):
            success, translated_yaml = translate_batch(
                yaml_input, source_name, target_name)
            if success:
                translated_data = yaml.safe_load(translated_yaml)
                translated_texts.extend(
                    entry['translation_text'] for entry in translated_data)
                consecutive_failures = 0
                break
            else:
                if attempt == max_retries - 1:
                    print(
                        f"Batch failed after {max_retries} attempts. Extracting partial translations...")
                    if translated_yaml:
                        partial_translations = extract_partial_translations(
                            translated_yaml)
                        for j in range(len(batch_subs)):
                            if j < len(partial_translations) and partial_translations[j]:
                                translated_texts.append(
                                    partial_translations[j])
                            else:
                                translated_texts.append(batch_subs[j].content)
                    else:
                        # API failure, no response available
                        translated_texts.extend(
                            sub.content for sub in batch_subs)
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print(
                            "Three consecutive batches failed. Stopping process.\ninput:\n{yaml_input}\noutput:\n{translated_yaml}")
                        return

    # Update subtitles with translations
    for sub, translated_text in zip(subs, translated_texts):
        sub.content = translated_text

    # Write output file
    output_path = args.output if args.output else args.input.replace(
        '.srt', f'_{args.target}.srt')
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))
        print(f"Translation complete. Output saved to '{output_path}'.")
    except Exception as e:
        print(f"Error writing output file: {e}")
        return


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitles using OpenAI API.")
    parser.add_argument("input", help="Path to input SRT file")
    parser.add_argument("--output", required=False, default=None,
                        help="Path to output translated SRT file")
    parser.add_argument("--source", required=False, default=None,
                        help="Source language code (e.g., 'en')")
    parser.add_argument("--target", required=False, default='zh',
                        help="Target language code (e.g., 'zh')")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of segments per API request")
    args = parser.parse_args()

    translate_srt(args)


if __name__ == "__main__":
    main()
