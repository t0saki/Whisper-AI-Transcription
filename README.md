# Whisper AI视频/语音文字转录工具

[English README](README_en.md) 

为了加速考前复习，写了一个工具，快速将视频/音频转化为文字资料，帮助节省反复观看视频的时间，提高复习效率。

## 主要功能

- 🎥 支持多种媒体格式（MP4/AVI/MKV/MOV/MP3/WAV等）
- ⏱️ 自动生成带时间戳的文字稿，方便快速定位重点
- 📝 一键生成纯文本转录文件
- 📜 可选生成SRT字幕文件（带精确时间戳）
- 🚀 支持GPU加速转录（自动检测可用设备）

## 快速使用

1. 安装依赖：安装[PyTorch](https://pytorch.org/get-started/locally/)，[transformers](https://huggingface.co/docs/transformers/installation)，[ffmpeg](https://ffmpeg.org/download.html)，[flash-attention](https://github.com/Dao-AILab/flash-attention)

2. 运行命令：
```bash
python whisper-cli.py [视频文件夹路径] [选项]

# 示例 - 生成文字稿和字幕
python whisper-cli.py ./lectures -l chinese -s
```

## 使用场景

✅ 考前快速整理讲座重点  
✅ 制作可搜索的课程笔记  
✅ 创建带时间戳的学习资料  
✅ 提取音频课程核心内容

## 注意事项

- 建议使用NVIDIA显卡或Apple M系列芯片获得最佳性能
- 首次运行会自动下载AI模型
- 确保已安装FFmpeg并加入系统路径