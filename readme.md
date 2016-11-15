# 环境 

VS2013 Ultimate x64 + OpenCV 2.4.11

Windows7 x64 下测试

# 代码说明

1. PanoStitch stitcher(".\\sample\\script.txt");
   参数为PTStitcher脚本所在路径。
   仅读入脚本中的p，i，o三种类型的行。
   目前不支持所有PTStitcher官方文档中标记为“obsolete”的选项，以及行中的p，i行对应的n参数，o行对应的C，S参数。
   其中o行对应的C，S参数将在之后添加支持。

2. stitcher.init(".\\sample\\", true, "avi");
   初始化拼接，第一个参数为视频所在路径，第三个参数为视频格式如“mp4”,将会从该路径读入名字为0,1,2,3..的视频，如“0.mp4”,"1.mp4"...
   需要保证天顶对应的视频位于视频序列的最后一个。
   第二个参数为bool值，设置为true表示视频目录下包含mask文件，将会从该路径读入名字为"mid_X.png"格式的8位图片，设置为false表示需要在视频路径下生成mask文件，
   将会在视频路径下生成"mid_X.png"文件。（注：目前生成mask过程较慢，约几分钟，所以在提供的两个例子中也提供了mask文件）。

3. stitcher.stitch(".\\sample\\res.avi");
   参数为输出的视频路径。

# 运行说明

  需要目录下所有的dll，特别注意，如果opencv_ffmpeg2411_64.dll不在运行环境中，不会报错，但是会导致无法正确生成视频。