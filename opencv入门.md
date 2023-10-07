---
typora-root-url: ./
---

# opencv入门

### **图像读取与显示**

头文件：

```c++
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
```



#### `imread()` 读取图像：

```c++
imread(const String & filename,
            int  flags=IMREAD_COLOR
            )
```

- filename：需要读取图像的文件名称，包含图像地址、名称和图像文件扩展名

- flags：读取图像形式的标志，如将彩色图像按照灰度图读取，默认参数是按照彩色图像格式读取，可选参数：

- | **标志参数**               | **简记** | **作用**                                                     |
  | -------------------------- | -------- | ------------------------------------------------------------ |
  | IMREAD_UNCHANGED           | -1       | 按照图像原样读取，保留Alpha通道（第4通道）                   |
  | IMREAD_GRAYSCALE           | 0        | 将图像转成单通道灰度图像后读取                               |
  | IMREAD_COLOR               | 1        | 将图像转换成3通道BGR彩色图像                                 |
  | IMREAD_ANYDEPTH            | 2        | 保留原图像的16位、32位深度，不声明该参数则转成8位读取        |
  | IMREAD_ANYCOLOR            | 4        | 以任何可能的颜色读取图像                                     |
  | IMREAD_LOAD_GDAL           | 8        | 使用gdal驱动程序加载图像                                     |
  | IMREAD_REDUCED_GRAYSCALE_2 | 16       | 将图像转成单通道灰度图像，尺寸缩小1/2，可以更改最后一位数字实现缩小1/4（最后一位改为4）和1/8（最后一位改为8） |
  | IMREAD_REDUCED_COLOR_2     | 17       | 将图像转成3通道彩色图像，尺寸缩小1/2，可以更改最后一位数字实现缩小1/4（最后一位改为4）和1/8（最后一位改为8） |
  | IMREAD_IGNORE_ORIENTATION  | 128      | 不以EXIF的方向旋转图像                                       |



#### `imshow()` 显示图像：

```c++
imshow(const String & winname,
                  InputArray mat
                  )
```

- winname：要显示图像的窗口的名字，用字符串形式赋值

- mat：要显示的图像矩阵

  

`src.empty()` 判断Mat类是否为空



#### `namedWindow()` 创建窗口：

```c++
namedWindow(const String & winname,
                        int  flags = WINDOW_AUTOSIZE
                        )
```

- winname：窗口名称，用作窗口的标识符

- flags：窗口属性设置标志

  

**`namedWindow()`函数窗口属性标志参数**

| **标志参数**        | **简记**   | **作用**                                 |
| ------------------- | ---------- | ---------------------------------------- |
| WINDOW_NORMAL       | 0x00000000 | 显示图像后，允许用户随意调整窗口大小     |
| WINDOW_AUTOSIZE     | 0x00000001 | 根据图像大小显示窗口，不允许用户调整大小 |
| WINDOW_OPENGL       | 0x00001000 | 创建窗口的时候会支持OpenGL               |
| WINDOW_FULLSCREEN   | 1          | 全屏显示窗口                             |
| WINDOW_FREERATIO    | 0x00000100 | 调整图像尺寸以充满窗口                   |
| WINDOW_KEEPRATIO    | 0x00000000 | 保持图像的比例                           |
| WINDOW_GUI_EXPANDED | 0x00000000 | 创建的窗口允许添加工具栏和状态栏         |
| WINDOW_GUI_NORMAL   | 0x00000010 | 创建没有状态栏和工具栏的窗口             |

**注意**

> 此函数运行后会继续执行后面程序，如果后面程序执行完直接退出的话，那么显示的图像有可能闪一下就消失了，因此在需要显示图像的程序中，往往会在`imshow()`函数后跟有`cv::waitKey()`函数，用于将程序暂停一段时间。`waitKey()`函数是以毫秒计的等待时长，如果参数缺省或者为“0”表示等待用户按键结束该函数。



#### 读取图像 代码：

```c++
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	Mat src = imread("newphoto.png", IMREAD_GRAYSCALE);//以灰度图打开

	if (src.empty())
	{
		printf("could not load image...\n");
		return -1;
	}

	namedWindow("input", WINDOW_FREERATIO);
	imshow("input", src);

	waitKey(0);
	destroyAllWindows;

	return 0;
}
```



### 视频文件与摄像头

#### 视频数据的读取

`VideoCapture（）`类构造函数：

```c++
cv :: VideoCapture :: VideoCapture(); //默认构造函数
cv :: VideoCapture :: VideoCapture(const String& filename,
                                         int apiPreference =CAP_ANY
                                         )
```

- `filename`：读取的视频文件或者图像序列名称
- `apiPreference`：读取数据时设置的属性，例如编码格式、是否调用OpenNI等

该函数是构造一个能够读取与处理视频文件的视频流，默认构造函数只是声明了一个能够读取视频数据的类，具体读取什么视频文件，需要在使用时通过open()函数指出，例如`cap.open(“1.avi”)是`VideoCapture类变量cap读取1.avi视频文件。

第二种构造函数在给出声明变量的同时也将视频数据赋值给变量。可以读取的文件种类包括视频文件(例如video.avi)、图像序列或者视频流的URL。其中读取图像序列需要将多个图像的名称统一为“前缀+数字”的形式，通过“前缀+%02d”的形式调用，例如在某个文件夹中有图片img_00.jpg、img_01.jpg、img_02.jpg……加载时文件名用img_%02d.jpg表示。函数中的读取视频设置属性标签默认的是自动搜索合适的标志，所以在平时使用中，可以将其缺省，只需要输入视频名称即可。与imread()函数一样，构造函数同样有可能读取文件失败，因此需要通过isOpened()函数进行判断，如果读取成功则返回值为true，如果读取失败，则返回值为false。



通过构造函数只是将视频文件加载到了VideoCapture类变量中，当我们需要使用视频中的图像时，还需要**将图像由VideoCapture类变量里导出到Mat类变量里**，用于后期数据处理，该操作可以通过“>>”运算符将图像按照视频顺序由VideoCapture类变量复制给Mat类变量。当VideoCapture类变量中所有的图像都赋值给Mat类变量后，再次赋值的时候Mat类变量会变为空矩阵，因此可以通过**empty()判断VideoCapture类变量中是否所有图像都已经读取完毕**。



VideoCapture类变量同时提供了可以查看视频属性的get()函数，通过输入指定的标志来获取视频属性，例如视频的像素尺寸、帧数、帧率等:

**VideoCapture类中get方法中的标志参数**

| **标志参数**          | **简记** | **作用**                           |
| --------------------- | -------- | ---------------------------------- |
| CAP_PROP_POS_MSEC     | 0        | 视频文件的当前位置（以毫秒为单位） |
| CAP_PROP_FRAME_WIDTH  | 3        | 视频流中图像的宽度                 |
| CAP_PROP_FRAME_HEIGHT | 4        | 视频流中图像的高度                 |
| CAP_PROP_FPS          | 5        | 视频流中图像的帧率（每秒帧数）     |
| CAP_PROP_FOURCC       | 6        | 编解码器的4字符代码                |
| CAP_PROP_FRAME_COUNT  | 7        | 视频流中图像的帧数                 |
| CAP_PROP_FORMAT       | 8        | 返回的Mat对象的格式                |
| CAP_PROP_BRIGHTNESS   | 10       | 图像的亮度（仅适用于支持的相机）   |
| CAP_PROP_CONTRAST     | 11       | 图像对比度（仅适用于相机）         |
| CAP_PROP_SATURATION   | 12       | 图像饱和度（仅适用于相机）         |
| CAP_PROP_HUE          | 13       | 图像的色调（仅适用于相机）         |
| CAP_PROP_GAIN         | 14       | 图像的增益（仅适用于支持的相机）   |

#### 代码实现

```c++
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    system("color F0"); //更改输出界面颜色
    VideoCapture video("cup.mp4");
    if (video.isOpened())
    {
        cout << "视频中图像的宽度=" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
        cout << "视频中图像的高度=" << video.get(CAP_PROP_FRAME_HEIGHT) << endl;
        cout << "视频帧率=" << video.get(CAP_PROP_FPS) << endl;
        cout << "视频的总帧数=" << video.get(CAP_PROP_FRAME_COUNT);
        }
    else
    {
        cout << "请确认视频文件名称是否正确" << endl;
        return -1;
    }
    while (1)
    {
        Mat frame;
        video >> frame;
        if (frame.empty())
        {
            break;
        }
        imshow("video", frame);
        waitKey(1000 / video.get(CAP_PROP_FPS));
    }
    waitKey();
    return 0;
}
```

#### 摄像头的直接调用

调用摄像头与读取视频文件相比，只有第一个参数不同。调用摄像头时，第一个参数为要打开的摄像头设备的ID，ID的命名方式从0开始。从摄像头中读取图像数据的方式与从视频中读取图像数据的方式相同，通过“>>”符号读取当前时刻相机拍摄到的图像。并且读取视频时VideoCapture类具有的属性同样可以使用。

```c++
	VideoCapture video;

	video.open(0);//打开摄像头

	if (!video.isOpened())
	{
		cout << "打开失败，请确认视频文件名称是否正确。" << endl;
		return -1;
	}

	cout << "视频中图像的宽度=" << video.get(CAP_PROP_FRAME_WIDTH);
	cout << "视频帧率=" << video.get(CAP_PROP_FPS);

	Mat img;
	video >> img;
	if (img.empty()) 
	{
		cout << "没有获取到图像" << endl;
		return -1;
	}

	bool isColor = (img.type() == CV_8UC3);//判断相机（视频）类型是否为彩色

	//直接打开视频
	/*imshow("video", img);
	int c = waitKey(1000 / video.get(CAP_PROP_FPS));//单位是毫秒

	if (c == 27) break;*/

	//保存视频
	VideoWriter writer;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');//选择编码格式

	double fps = 25.0;//设置视频帧率，比要保存视频的帧率大，则就是快放；否则，就是慢放

	string filename = "video.avi";//保存视频文件的名称
	writer.open(filename, codec, fps, img.size(), isColor);//创建保存视频文件的视频流

	if (!writer.isOpened())//判断视频流是否创建成功
	{
		cout << "打开视频文件失败，请确认是否为合法输入" << endl;
		return -1;
	}

	while (1)
	{
		//检测是否执行完毕
		if (!video.read(img))//判断能否继续从摄像头获视频文件中读出一帧图像
		{
			cout << "摄像头断开连接或者视频读取完成" << endl;
			break;
		}
		writer.write(img);//把图像写入视频流
		                  //writer << img
		imshow("Live", img);//显示图像
		char c = waitKey(50);
		if (c == 27) //按ESC退出视频保存
		{
			break;
		}
	}

	//退出程序时自动关闭视频流
	//vedio.release()
	//writer.release()
```



### 图像、视频保存

#### `imwrite()`保存图像

```c++
imwrite(const String& filename,
        InputArray img,
        Const std::vector<int>& params = std::vector<int>()
        )
```

- filename：保存图像的地址和文件名，包含图像格式
- img：将要保存的Mat类矩阵变量
- params：保存图片格式属性设置标志

该函数用于将Mat类矩阵保存成图像文件，如果成功保存，则返回true，否则返回false。可以保存的图像格式参考imread()函数能够读取的图像文件格式，通常使用该函数只能保存8位单通道图像和3通道BGR彩色图像，但是可以通过更改第三个参数保存成不同格式的图像。不同图像格式能够保存的图像位数如下：

- 16位无符号（CV_16U）图像可以保存成PNG、JPEG、TIFF格式文件；
- 32位浮点（CV_32F）图像可以保存成PFM、TIFF、OpenEXR和Radiance HDR格式文件；
- 4通道（Alpha通道）图像可以保存成PNG格式文件。



#### `VideoWrite()`视频保存

```c++
读取视频文件VideoCapture类构造函数
cv :: VideoWriter :: VideoWriter(); //默认构造函数
cv :: VideoWriter :: VideoWriter(const String& filename,
                                       int fourcc,
                                       double  fps,
                                       Size frameSize,
                                       bool  isColor=true
                                       )
```

- filename：保存视频的地址和文件名，包含视频格式
- int：压缩帧的4字符编解码器代码，详细参数在表2-7给出。
- fps：保存视频的帧率，即视频中每秒图像的张数。
- framSize：视频帧的尺寸
- isColor：保存视频是否为彩色视频

第1行默认构造函数的使用方法与VideoCapture()相同，都是创建一个用于保存视频的数据流，后续通过open()函数设置保存文件名称、编解码器、帧数等一系列参数。第二种构造函数需要输入的第一个参数是需要保存的视频文件名称，第二个函数是编解码器的代码，可以设置的编解码器选项在表中给出，如果赋值“-1”则会自动搜索合适的编解码器，需要注意的是其在OpenCV 4.0版本和OpenCV 4.1版本中的输入方式有一些差别。第三个参数为保存视频的帧率，可以根据需求自由设置，例如实现原视频二倍速播放、原视频慢动作播放等。第四个参数是设置保存的视频文件的尺寸，这里需要注意的时，在设置时一定要与图像的尺寸相同，不然无法保存视频。最后一个参数是设置保存的视频是否是彩色的，程序中，默认的是保存为彩色视频。

该函数与VideoCapture()有很大的相似之处，都可以通过isOpened()函数判断是否成功创建一个视频流，可以通过get()查看视频流中的各种属性。在保存视频时，我们只需要将生成视频的图像一帧一帧通过“<<”操作符（或者write()函数）赋值给视频流即可，最后使用release()关闭视频流。



### 图像色彩空间、数据类型转换

#### 图像色彩模式：

BGR模式，HSV模式（色调H，饱和度S，亮度V），位图模式，灰度图模式

#### `cvtColor()`色彩空间转换函数：

```c++
cvtColor	(	InputArray src,
                OutputArray dst,
                       int code,
                       int dstCn = 0 
                      )
```

第一个参数为输入图像;
第二个参数为输出图像;
第三个参数为颜色空间转换的标识符（具体见表）;
第四个参数为目标图像的通道数，若该参数是0，表示目标图像取源图像的通道数。

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231002170819847.png" alt="image-20231002170819847" style="zoom:33%;" />

#### 代码实现：

```c++
Mat gray, hsv;

cvtColor(src, hsv, COLOR_BGR2HSV);
cvtColor(src, gray, COLOR_BGR2GRAY);

imshow("input", src);
imshow("HSV", hsv);
imshow("GRAY", gray);
```



#### inRange()函数

OpenCV中的函数inRange()用于将指定值范围的像素选出来。如果像素的值满足指定的范围，则这个像素点的值被置为255，否则值被置为0。

其函数原型如下：

```c++
void cv::inRange(	InputArray 	src,
					InputArray 	lowerb,
					InputArray 	upperb,
					OutputArray dst 
				)
```

```c++
Mat hsv;
cvtColor(image, hsv, COLOR_BGR2HSV);
Mat mask;
inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
imshow("mask1", mask);
Mat redback = Mat::zeros(image.size(), image.type());
redback = Scalar(40, 40, 200);
bitwise_not(mask, mask);
imshow("原图", image);
imshow("mask2", mask);
image.copyTo(redback, mask);
imshow("roi区域提取", redback);
```

#### `image.copyTo()`有两种形式：

1、`image.copyTo(imageROI)`，作用是把image的内容粘贴到imageROI；

2、`image.copyTo(imageROI，mask)`,作用是把mask和image重叠传递给`imageRoi`



#### `convertTo()`数据类型转换函数

```c++
convertTo(OutputArry m,
          		int rtype,
                double alpha = 1,
                double beta = 0
                )
```

- m：转换类型后输出的图像。
- rtype：转换图像的数据类型。
- alpha：转换过程中的缩放因子。
- beta：转换过程中的偏置因子。

该函数用来实现将已有图像转换成指定数据类型的图像，第一个参数用于输出转换数据类型后的图像，第二个参数用于声明转换后图像的数据类型。第三个与第四个参数用于声明两个数据类型间的转换关系，具体转换形式如式所示。

![图片](D:\wsq\课程文件\计算机\视觉组\640.png)

通过转换公式可以知道该转换方式就是将原有数据进行线性转换，并按照指定的数据类型输出。根据其转换规则可以知道，该函数不仅能够实现不同数据类型之间的转换，还能实现在同一种数据类型中的线性变换。



#### 代码实现

为了防止转换后出现数值越界的情况，先将CV_8U类型转成CV_32F类型后再进行颜色模型的转换。

```c++
Mat gray, HSV, YUV, Lab, img32;
img.convertTo(img32, CV_32F, 1.0 / 255); //将CV_8U类型转换成CV_32F类型
//img32.convertTo(img, CV_8U, 255); //将CV_32F类型转换成CV_8U类型
cvtColor(img32, HSV, COLOR_BGR2HSV);
cvtColor(img32, YUV, COLOR_BGR2YUV);
cvtColor(img32, Lab, COLOR_BGR2Lab);
cvtColor(img32, gray, COLOR_BGR2GRAY);
imshow("原图", img32);
imshow("HSV", HSV);
imshow("YUV", YUV);
imshow("Lab", Lab);
imshow("gray", gray);
```



### Mat类创建

Mat类赋值时是把指针指向了赋值的数据块，浅拷贝；

只有克隆或拷贝时，Mat类才指向复制的另一块数据，深拷贝。

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231002174558062.png" alt="image-20231002174558062" style="zoom: 50%;" />

#### 创建Mat类

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231002174826629.png" alt="image-20231002174826629" style="zoom: 67%;" />

最后一种适合快速创建卷积核



#### **OpenCV中的数据类型与取值范围**

| **数据类型** | **具体类型**   | **取值范围**               |
| ------------ | -------------- | -------------------------- |
| CV_8U        | 8位无符号整数  | 0—255                      |
| CV_8S        | 8位符号整数    | -128—127                   |
| CV_16U       | 16位无符号整数 | 0-65535                    |
| CV_16S       | 16位符号整数   | -32768—32767               |
| CV_32S       | 32位符号整数   | -2147483648—2147483647     |
| CV_32F       | 32位浮点整数   | -FLT_MAX—FLT_MAX, INF, NAN |
| CV_64F       | 64位浮点整数   | -DBL_MAX—DBL_MAX, INF, NAN |

通道数标识：C1、C2、C3、C4分别表示单通道、双通道、3通道和4通道

```c++
Mat a(640,480,CV_8UC3) *//创建一个640\*480的3通道矩阵用于存放彩色图像*
Mat a(3,3,CV_8UC1) *//创建一个3\*3的8位无符号整数的单通道矩阵*
Mat a(3,3,CV_8U) *//创建单通道矩阵C1标识可以省略*
```






#### Mat类构造

##### **（1）利用默认构造函数**

```c++
cv::Mat::Mat();
```

这种构造方式不需要输入任何的参数，在后续给变量赋值的时候会自动判断矩阵的类型与大小，实现灵活的存储，常用于存储读取的图像数据和某个函数运算输出结果。

##### **（2）根据输入矩阵尺寸和类型构造**

```c++
cv::Mat::Mat( int  rows,
           int  cols,
           int  type
          )
```

- rows：构造矩阵的行数
- cols：矩阵的列数
- type：矩阵中存储的数据类型。此处除了CV_8UC1、CV_64FC4等从1到4通道以外，还提供了更多通道的参数，通过CV_8UC(n)中的n来构建多通道矩阵，其中n最大可以取到512.

通过输入矩阵的行、列以及存储数据类型实现构造。这种定义方式清晰、直观、易于阅读，常用在明确需要存储数据尺寸和数据类型的情况下，例如相机的内参矩阵、物体的旋转矩阵等。利用输入矩阵尺寸和数据类型构造Mat类的方法存在一种变形，通过将行和列组成一个Size()结构进行赋值:

```c++
用Size()结构构造Mat类
cv::Mat::Mat(Size size(),
               int  type
               )
```

- size：2D数组变量尺寸，通过Size(cols, rows)进行赋值。

利用这种方式构造Mat类时要格外注意，在Size()结构里矩阵的行和列的顺序与代码清单2-5中的方法相反，使用Size()时，列在前、行在后。如果不注意同样会构造成功Mat类，但是当我们需要查看某个元素时，我们并不知道行与列颠倒，就会出现数组越界的错误。使用该种方法构造函数如下：

```c++
用Size()结构构造Mat示例

cv::Mat a(Size(480, 640), CV_8UC1); //构造一个行为640，列为480的单通道矩阵

cv::Mat b(Size(480, 640), CV_32FC3); //构造一个行为640，列为480的3通道矩
```

##### **（3）利用已有矩阵构造**

```c++
cv::Mat::Mat( const Mat & m);

m：已经构建完成的Mat类矩阵数据。
```

这种构造方式非常简单，可以构造出与已有的Mat类变量存储内容一样的变量。注意**这种构造方式只是复制了Mat类的矩阵头，矩阵指针指向的是同一个地址，因此如果通过某一个Mat类变量修改了矩阵中的数据，另一个变量中的数据也会发生改变。**

**复制两个一模一样的Mat类而彼此之间不会受影响**:

```c++
Mat m1 = src.clone();//克隆

Mat m2;
src.copyTo(m2);//复制
```



##### （4）构造已有Mat类的子类

**如果需要构造的矩阵尺寸比已有矩阵小，并且存储的是已有矩阵的子内容：**

```c++
构造已有Mat类的子类
cv::Mat::Mat(const Mat & m,
               const Range & rowRange,
               const Range & colRange = Range::all()
               )
```

- m：已经构建完成的Mat类矩阵数据。
- rowRange：在已有矩阵中需要截取的行数范围，是一个Range变量，例如从第2行到第5行可以表示为Range(2,5)。
- colRange：在已有矩阵中需要截取的列数范围，是一个Range变量，例如从第2列到第5列可以表示为Range(2,5)，当不输入任何值时表示所有列都会被截取。

这种方式主要用于在原图中截图使用，不过需要注意的是，**通过这种方式构造的Mat类与已有Mat类享有共同的数据**，即如果两个Mat类中有一个数据发生更改，另一个也会随之更改。

使用该种方法构造Mat类如图：

```c++
cv::Mat a(
		A,
		Range(1,3),//截取行
		Range(1,3)//截取列
		);
```



#### Mat类赋值

##### （1）构造时赋值

```c++
cv::Mat::Mat(int  rows,
               int  cols,
               int  type,
               const Scalar & s
               )
```

s：**给矩阵中每个像素赋值的参数变量**，例如Scalar(0, 0, 255)。

该种方式是在构造的同时进行赋值，将每个元素想要赋予的值放入Scalar结构中即可，这里需要注意的是，用此方法会将图像中的每个元素赋值相同的数值，例如Scalar(0, 0, 255)会将每个像素的三个通道值分别赋值0，0，255。

```c++
cv::Mat a(2, 2, CV_8UC3, cv::Scalar(0,0,255));//创建一个3通道矩阵，每个像素都是0，0，255
```

**提示**

Scalar结构中变量的个数一定要与定义中的通道数相对应，如果Scalar结构中变量个数大于通道数，则位置大于通道数之后的数值将不会被读取，例如执行a(2, 2, CV_8UC2, Scalar(0,0,255))后，每个像素值都将是（0,0），而255不会被读取。如果Scalar结构中变量数小于通道数，则会以0补充。



##### **（2）枚举赋值法**

这种赋值方式是将矩阵中所有的元素都一一枚举出，并用数据流的形式赋值给Mat类。

```c++
cv::Mat a = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
cv::Mat b = (cv::Mat_<double>(2, 3) << 1.0, 2.1, 3.2, 4.0, 5.1, 6.2);
```

上面第一行代码创建了一个3×3的矩阵，矩阵中存放的是从1-9的九个整数，先将矩阵中的第一行存满，之后再存入第二行、第三行，即1、2、3存放在矩阵a的第一行，4、5、6存放在矩阵a的第二行，7，8，9存放在矩阵a的第三行。第二行代码创建了一个2×3的矩阵，其存放方式与矩阵a相同。

**提示**

采用枚举法时，输入的数据个数一定要与矩阵元素个数相同，例如上图第一行代码只输入从1到8八个数，赋值过程会出现报错，因此本方法常用在矩阵数据比较少的情况。



##### **（3）循环赋值**

与通过枚举法赋值方法相类似，循环法赋值也是对矩阵中的每一位元素进行赋值，但是可以不在声明变量的时候进行赋值，而且可以对矩阵中的任意部分进行赋值。

```c++
cv::Mat c = cv::Mat_<int>(3, 3); //定义一个3*3的矩阵

for (int i = 0; i < c.rows; i++) //矩阵行数循环
{
	for (int j = 0; j < c.cols; j++) //矩阵列数循环
		{
			c.at<int>(i, j) = i+j;
		}
}
```



上面代码同样创建了一个3×3的矩阵，通过for循环的方式，对矩阵中的每一位元素进行赋值。需要注意的是，在给矩阵每个元素进行赋值的时候，赋值函数中声明的变量类型要与矩阵定义时的变量类型相同，即上面代码中第1行和第6行中变量类型要相同，如果第6行代码改成c.at<double>(i, j) ，程序就会报错，无法赋值。



##### **（4）类方法赋值**

在Mat类里提供了可以快速赋值的方法，可以初始化指定的矩阵。例如生成单位矩阵、对角矩阵、所有元素都为0或者1的矩阵等。

```c++
cv::Mat a = cv::Mat::eye(3, 3, CV_8UC1);
cv::Mat b = (cv::Mat_<int>(1, 3) << 1, 2, 3);
cv::Mat c = cv::Mat::diag(b);
cv::Mat d = cv::Mat::ones(3, 3, CV_8UC1);
cv::Mat e = cv::Mat::zeros(4, 2, CV_8UC3);
```

上面代码中，每个函数作用及参数含义分别如下：

- eye()：构建一个单位矩阵，前两个参数为矩阵的行数和列数，第三个参数为矩阵存放的数据类型与通道数。如果行和列不相等，则在矩阵的 (1,1)，(2,2)，(3,3)等主对角位置处为1。

- diag()：构建对角矩阵，其参数必须是Mat类型的1维变量，用来存放对角元素的数值。

- ones()：构建一个全为1的矩阵，参数含义与eye()相同。

- zeros()：构建一个全为0的矩阵，参数含义与eye()相同。

  

##### **（5）利用数组进行赋值**

这种方法与枚举法相类似，但是该方法可以根据需求改变Mat类矩阵的通道数，可以看作枚举法的拓展，

```c++
float a[8] = { 5,6,7,8,1,2,3,4 };
cv::Mat b = cv::Mat(2, 2, CV_32FC2, a);
cv::Mat c = cv::Mat(2, 4, CV_32FC1, a);
```

这种赋值方式首先将需要存入到Mat类中的变量存入到一个数组中，之后通过设置Mat类矩阵的尺寸和通道数将数组变量拆分成矩阵，这种拆分方式可以自由定义矩阵的通道数，当矩阵中的元素数目大于数组中的数据时，将用-1.0737418e+08填充赋值给矩阵，如果矩阵中元素的数目小于数组中的数据时，将矩阵赋值完成后，数组中剩余数据将不再赋值。由数组赋值给矩阵的过程是首先将矩阵中第一个元素的所有通道依次赋值，之后再赋值下一个元素。



### 读写Mat类

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133581-1.png)

​                                                                                           三通道3*3矩阵存储方式

#### **Mat类矩阵的常用属性**

| **属性**   | **作用**                     |
| ---------- | ---------------------------- |
| cols       | 矩阵的列数                   |
| rows       | 矩阵的行数                   |
| step       | 以字节为单位的矩阵的有效宽度 |
| elemSize() | 每个元素的字节数             |
| total()    | 矩阵中元素的个数             |
| channels() | 矩阵的通道数                 |



#### 读取方法

##### 通过at方法读取Mat类矩阵中的元素

通过at方法读取矩阵元素分为针对单通道的读取方法和针对多通道的读取方法，在代码清单2-19中给出了通过at方法读取单通道矩阵元素的代码。

```c++
cv::Mat a = (cv::Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
int value = (int)a.at<uchar>(0, 0);
```

通过at方法读取元素需要在后面跟上“<数据类型>”，如果此处的数据类型与矩阵定义时的数据类型不相同，就会出现因数据类型不匹配的报错信息。该方法以坐标的形式给出需要读取的元素坐标(行数，列数)。需要说明的是，如果矩阵定义的是uchar类型的数据，在需要输入数据的时候，需要强制转换成int类型的数据进行输出，否则输出的结果并不是整数。

由于单通道图像是一个二维矩阵，因此在at方法的最后给出二维平面坐标即可访问对应位置元素。

而多通道矩阵每一个元素坐标处都是多个数据，因此引入一个变量用于表示同一元素多个数据。

在openCV 中，针对3通道矩阵，定义了cv::Vec3b、cv::Vec3s、cv::Vec3w、cv::Vec3d、cv::Vec3f、cv::Vec3i六种类型用于表示同一个元素的三个通道数据。通过这六种数据类型可以总结出其命名规则，其中的数字表示通道的个数，最后一位是数据类型的缩写，b是uchar类型的缩写、s是short类型的缩写、w是ushort类型的缩写、d是double类型的缩写、f是float类型的缩写、i是int类型的缩写。

当然OpenCV也为2通道和4通道定义了对应的变量类型，其命名方式也遵循这个命名规则，例如2通道和4通道的uchar类型分别用cv::Vec2b和cv::Vec4b表示。代码清单2-20中给出了通过at方法读取多通道矩阵的实现代码。

```c++
cv::Mat b(3, 4, CV_8UC3, cv::Scalar(0, 0, 1));
cv::Vec3b vc3 = b.at<cv::Vec3b>(0, 0);
int first = (int)vc3.val[0];
int second = (int)vc3.val[1];
int third = (int)vc3.val[2];
```

在使用多通道变量类型时，同样需要注意at方法中数据变量类型与矩阵的数据变量类型相对应，并且cv::Vec3b类型在输入每个通道数据时需要将其变量类型强制转成int类型。不过，如果直接将at方法读取出的数据直接赋值给cv::Vec3i类型变量，就不需要在输出每个通道数据时进行数据类型的强制转换。



##### 通过指针ptr读取Mat类矩阵中的元素

```c++
cv::Mat b(3, 4, CV_8UC3, cv::Scalar(0, 0, 1));
for (int i = 0; i < b.rows; i++)
{
	uchar* ptr = b.ptr<uchar>(i);
	for (int j = 0; j < b.cols*b.channels(); j++)
	{
		cout << (int)ptr[j] << endl;
	}
}
```

当我们能够确定需要访问的数据时，可以直接通过给出行数和指针后移的位数进行访问，例如当读取第2行数据中第3个数据时，可以用`a.ptr<uchar>(1)[2]`



##### 通过迭代器访问Mat类矩阵中的元素

Mat类变量同时也是一个容器变量，所以Mat类变量拥有迭代器，用于访问Mat类变量中的数据，通过迭代器可以实现对矩阵中每一个元素的遍历，代码实现在代码清单2-22中给出。

```c++
cv::MatIterator_<uchar> it = a.begin<uchar>();
cv::MatIterator_<uchar> it_end = a.end<uchar>();
for (int i = 0; it != it_end; it++)
{
	cout << (int)(*it) << " ";
	if ((++i% a.cols) == 0)
	{
		cout << endl;
	}
}
```

Mat类的迭代器变量类型是`cv::MatIterator_< >`，在定义时同样需要在括号中声明数据的变量类型。Mat类迭代器的起始是`Mat.begin< >()`，结束是`Mat.end< >()`，与其他迭代器用法相同，通过“++”运算实现指针位置向下迭代，数据的读取方式是先读取第一个元素的每一个通道，之后再读取第二个元素的每一个通道，直到最后一个元素的最后一个通道。



##### 通过矩阵元素地址定位方式访问元素

前面三种读取元素的方式都需要知道Mat类矩阵存储数据的类型，而且在从认知上，我们更希望能够通过声明“第x行第x列第x通道”的方式来读取某个通道内的数据，代码清单2-23中给出的就是这种读取数据的方式。

```c++
(int)(*(b.data + b.step[0] * row + b.step[1] * col + channel));
```



##### 代码实现

```c++
Mat src = image.clone();

	int w = src.cols;
	int h = src.rows;
	int dims = src.channels();

	//at法
	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)//灰度图像
			{
				int pv = src.at<uchar>(row, col);
				src.at<uchar>(row, col) = 255 - pv;
			}
			if (dims == 3)
			{
				Vec3b bgr = src.at<Vec3b>(row, col);
				src.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				src.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				src.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}

	//指针法
	for (int row = 0; row < h; row++)
	{
		uchar* current_row = src.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			if (dims == 1)
			{
				int pv = *current_row;
				*current_row++ = 255 - pv;
			}
			if (dims == 3)
			{
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}

	imshow("原图", image);
	imshow("改后图", src);
```



### Mat类的运算

Mat类支持加减乘除：

```c++
cv::Mat a = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
cv::Mat b = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
cv::Mat c = (cv::Mat_<double>(3, 3) << 1.0, 2.1, 3.2, 4.0, 5.1, 6.2, 2, 2, 2);
cv::Mat d = (cv::Mat_<double>(3, 3) << 1.0, 2.1, 3.2, 4.0, 5.1, 6.2, 2, 2, 2);
cv::Mat e, f, g, h, i;
e = a + b;
f = c - d;
g = 2 * a;
h = d / 2.0;
i = a – 1;
```

当两个类进行加减运算时，需保证数据要一致，比如int和double类型数据的两个类不能进行加减运算。常数与Mat类变量运算，结果的数据类型保留Mat类变量的数据类型。



也支持与Scalar()进行运算：

```c++
Mat m1,m2;
m2 = m1 + Scalar(50,50);
m2 = m1 - Scalar(50,50);
m2 = m1 * Scalar(2,2);
m2 = m1 / Scalar(2,2);
```



两个Mat类进行加减乘除可以用opencv内置函数：

```c++
//m1,m2是进行运算的Mat类，m3是输出结果
add(m1,m2,m3);//加法

subtract(m1,m2,m3);//减法

divide(m1,m2,m3);//除法

//乘法：

double k;

e = c*d;//数学乘积

k = a.dot(b);//内积

f=a.mul(b)//对应位乘积
```



**saturate_cast<>()**

saturate_cast<uchar>主要是为了防止颜色溢出操作=>截断操作

```c++
原理大致如下:

if(data<0) 

        data=0; 

elseif(data>255) 

    data=255;
```



### 滚动条操作

#### createTrackbar()函数

```c++
int cv::createTrackbar(const String & trackbarname,
                       		const String & winname,
                       		int * value,
                       		int  count,
                       		TrackbarCallback onChange = 0,
                      	 	void * userdata = 0 
                       		)
```

- trackbarname：滑动条的名称
- winname：创建滑动条窗口的名称。
- value：指向整数变量的指针，该指针指向的值反映滑块的位置，创建后，滑块位置由此变量定义。
- count：滑动条的最大取值。
- onChange：每次滑块更改位置时要调用的函数的指针。该函数应该原型为void Foo（int，void *）;，其中第一个参数是轨迹栏位置，第二个参数是用户数据。如果回调是NULL指针，则不会调用任何回调，只更新数值。
- userdata：传递给回调函数的可选参数

该函数能够在图像窗口的上方创建一个范围从0开始的整数滑动条，由于滑动条只能输出整数，如果需要得到小数，必须进行后续处理，例如输出值除以10得到含有1位小数的数据。

函数第一个参数是滑动条的名称，第二个参数是创建滑动条的图像窗口的名称。第三个参数是指向整数变量的指针，该指针指向的值反映滑块的位置，在创建滑动条时该参数确定了滑动块的初始位置，当滑动条创建完成后，该指针指向的整数随着滑块的移动而改变。第四个参数是滑动条的最大取值。第五个参数是每次滑块更改位置时要调用的函数的指针。

该函数应该原型为void Foo（int，void *），其中第一个参数是轨迹栏位置，第二个参数是用户数据，如果回调是NULL指针，则不会调用任何回调，只更新数值。最后一个参数是传递给回调函数的void *类型数据，如果使用的第三个参数是全局变量，可以不用忽略最后一个参数，使用参数的默认值即可。



#### 代码实现

```c++
static void on_lightness(int b, void* userdata)
{
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1.0, m, 0.0, b, dst);
	imshow("亮度与对比度调整", dst);
}

static void on_constrast(int b, void* userdata)
{
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);
	imshow("亮度与对比度调整", dst);
}

int main(int argc, char* argv[])
{
	Mat image = imread("newphoto.png");

	if (image.empty())
	{
		printf("could not load image...\n");
		return -1;
	}

	namedWindow("亮度与对比度调整", WINDOW_AUTOSIZE);
	int lightness = 50;
	int max_value = 100;
	int contrast_value = 100;
	createTrackbar("Value Bar", "亮度与对比度调整", &lightness, max_value, on_lightness, (void*)(&image));
	createTrackbar("Contrast Bar", "亮度与对比度调整", &contrast_value, 200, on_constrast, (void*)(&image));
```



### 键盘响应操作

`waitKey()`函数：

cv.waitKey( [, delay] ) --> retval
delay如果没有声明或者delay=0,表示一直阻塞
delay大于0，表示阻塞指定毫秒数
retval返回的对应键盘键值，注意:在不同的操作系统中可能会有差异！
典型的retval = 27是ESC按键（windows环境下）(数字与字母键按照对应的ASCII码)

```c++
char c = waitKey(50);
		if (c == 27) //按ESC退出视频保存
		{
			break;
		}
```

```c++
if(c == 27)
{
    break;
}
else if(c == 49)
{ 
    cout << "you enter key # 1" << endl;
    cvtColor(image, dst, COLOR_BGR2GRAY)
}
else if ...
```



### 颜色表操作

#### LUT查找表：

LUT 是 LookUpTable 的简称，也称作颜色查找表

从名字上看，我们大概可以知道，是用来查找颜色的，他确实就是这样的，是通过一种颜色，查找其映射后的颜色，可以理解为一个函数LUT(R1,G1,B1),带R,G,B三个自变量的函数，输出为其对应映射后的值R2,G2,B2

`LUT(R1, G1, B1) = (R2, G2, B2)`



LUT查找表简单来说就是一个像素灰度值的映射表，它以像素灰度值作为索引，以灰度值映射后的数值作为表中的内容。

例如我们有一个长度为5的存放字符的数组，LUT查找表就是通过这个数组将0映射成*a*，将1映射成*b*，依次类推。在OpenCV 4中提供了LUT()函数用于实现图像像素灰度值的LUT查找表功能：

```c++
void cv::LUT(InputArray src,
            	 InputArray lut,
            	 OutputArray dst
          		  )
```

- src：输入图像矩阵，其数据类型只能是CV_8U。
- lut：256个像素灰度值的查找表，单通道或者与src通道数相同。
- dst：输出图像矩阵，其尺寸与src相同，数据类型与lut相同。

该函数的第一个输入参数要求的数据类型必须是CV_8U类型，但是可以是多通道的图像矩阵。

第二个参数根据其参数说明可以知道输入量是一个1×256的矩阵，其中存放着每个像素灰度值映射后的数值.

如果第二个参数是单通道，则输入变量中的每个通道都按照一个LUT查找表进行映射；如果第二个参数是多通道，则输入变量中的第i个通道按照第二个参数的第i个通道LUT查找表进行映射。

与之前的函数不同，函数输出图像的数据类型不与原图像的数据类型保持一致，而是和LUT查找表的数据类型保持一致，这是因为将原灰度值映射到新的空间中，因此需要与新空间中的数据类型保持一致。

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133581-2.png)

 LUT查找表设置示例

```c++
//LUT查找表第一层
uchar lutFirst[256];
for (int i = 0; i<256; i++)
{
  if (i <= 100)
    lutFirst[i] = 0;
  if (i > 100 && i <= 200)
    lutFirst[i] = 100;
  if (i > 200)
    lutFirst[i] = 255;
}
Mat lutOne(1, 256, CV_8UC1, lutFirst);

//LUT查找表第二层
uchar lutSecond[256];
for (int i = 0; i<256; i++)
{
  if (i <= 100)
    lutSecond[i] = 0;
  if (i > 100 && i <= 150)
    lutSecond[i] = 100;
  if (i > 150 && i <= 200)
    lutSecond[i] = 150;
  if (i > 200)
    lutSecond[i] = 255;
}
Mat lutTwo(1, 256, CV_8UC1, lutSecond);

//LUT查找表第三层
uchar lutThird[256];
for (int i = 0; i<256; i++)
{
  if (i <= 100)
    lutThird[i] = 100;
  if (i > 100 && i <= 200)
    lutThird[i] = 200;
  if (i > 200)
    lutThird[i] = 255;
}
Mat lutThree(1, 256, CV_8UC1, lutThird);

//拥有三通道的LUT查找表矩阵
vector<Mat> mergeMats;
mergeMats.push_back(lutOne);
mergeMats.push_back(lutTwo);
mergeMats.push_back(lutThree);
Mat LutTree;
merge(mergeMats, LutTree);

//计算图像的查找表
Mat img = imread("lena.png");
if (img.empty())
{
  cout << "请确认图像文件名称是否正确" << endl;
  return -1;
}

Mat gray, out0, out1, out2;
cvtColor(img, gray, COLOR_BGR2GRAY);
LUT(gray, lutOne, out0);
LUT(img, lutOne, out1);
LUT(img, LutTree, out2);
imshow("out0", out0);
imshow("out1", out1);
imshow("out2", out2);
```

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133581-3.jpeg)



#### `applyColorMap()伪色彩函数`





![image-20231004173220749](D:\wsq\课程文件\计算机\视觉组\image-20231004173220749.png)

![image-20231003111933274](D:\wsq\课程文件\计算机\视觉组\image-20231003111933274.png)

### 图像像素逻辑操作

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003125037636.png" alt="image-20231003125037636" style="zoom: 50%;" />

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003125205685.png" alt="image-20231003125205685" style="zoom:50%;" />

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003125216036.png" alt="image-20231003125216036" style="zoom:50%;" />

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003125232403.png" alt="image-20231003125232403" style="zoom:50%;" />

### 通道分离与合并

#### 多通道分离函数split()

OpenCV 4中针对多通道分离函数split()有两种重载原型

```c++
 void cv::split(const Mat & src,
                Mat * mvbegin
                )
 
 void cv::split(InputArray m,
                OutputArrayOfArrays mv
                )
```

- src：待分离的多通道图像。

- mvbegin：分离后的单通道图像，为数组形式，数组大小需要与图像的通道数相同

- m：待分离的多通道图像

- mv：分离后的单通道图像，为向量vector形式

  

该函数主要是用于将多通道的图像分离成若干单通道的图像，两个函数原型中不同之处在于前者第二个参数输入的是Mat类型的数组，其数组的长度需要与多通道图像的通道数相等并且提前定义；

第二种函数原型的第二个参数输入的是一个vector<Mat>容器，不需要知道多通道图像的通道数。两个函数原型虽然输入参数的类型不同，但是通道分离的原理是相同的

#### 多通道合并函数merge()

OpenCV 4中针对多通道合并函数merge ()也有两种重载原型

```c++
 void cv::merge(const Mat * mv,
                 size_t  count,
                 OutputArray dst
                )
 
 void cv::merge(InputArrayOfArrays mv,
                 OutputArray dst
                )
```

- mv：需要合并的图像数组，其中每个图像必须拥有相同的尺寸和数据类型。

- count：输入的图像数组的长度，其数值必须大于0.

- mv：需要合并的图像向量vector，其中每个图像必须拥有相同的尺寸和数据类型。

- dst：合并后输出的图像，与mv[0]具有相同的尺寸和数据类型，通道数等于所有输入图像的通道数总和。

  

该函数主要是用于将多个图像合并成一个多通道图像，该函数也具有两种不同的函数原型，每一种函数原型都是与split()函数像对应，两种原型分别输入数组形式的图像数据和向量vector形式的图像数据，在输入数组形式数据的原型中，还需要输入数组的长度。

合并函数的输出结果是一个多通道的图像，其通道数目是所有输入图像通道数目的总和。

这里需要说明的是，用于合并的图像并非都是单通道的，也可以是多个通道数目不相同的图像合并成一个通道更多的图像，虽然这些图像的通道数目可以不相同，但是需要所有图像具有相同的尺寸和数据类型。



#### 颜色通道交换函数mixChannels()

```c++
void cv::mixChannels	(	
	InputArrayOfArrays 	src,
	InputOutputArrayOfArrays 	dst,
	const std::vector< int > & 	fromTo 
	)
```

第一个参数：输入矩阵
第二个参数：输出矩阵
第三个参数：复制列表，表示第输入矩阵的第几个通道复制到输出矩阵的第几个通道
比如 {0,2,1,1,2,0}表示：
src颜色通道0复制到dst颜色通道2
src颜色通道1复制到dst颜色通道1
src颜色通道2复制到dst颜色通道0



#### 代码实现

```c++
vector<Mat> mv;
split(image, mv);
imshow("蓝色", mv[0]);
imshow("绿色", mv[1]);
imshow("红色", mv[2]);

Mat dst;
mv[0] = 0;
merge(mv, dst);
imshow("红色", dst);

int from_to[] = { 0,2,1,1,2,0 };
mixChannels(image, dst, from_to, 3);//颜色通道交换
imshow("交换", dst);
```

```c++
//输入数组参数的多通道分离与合并
 Mat imgs[3];
 split(img, imgs);
 imgs0 = imgs[0];
 imgs1 = imgs[1];
 imgs2 = imgs[2];
 imshow("RGB-R通道", imgs0); //显示分离后R通道的像素值
 imshow("RGB-G通道", imgs1); //显示分离后G通道的像素值
 imshow("RGB-B通道", imgs2); //显示分离后B通道的像素值
 imgs[2] = img; //将数组中的图像通道数变成不统一
 merge(imgs, 3, result0); //合并图像
 //imshow("result0", result0); //imshow最多显示4个通道，因此结果在Image Watch中查看
 Mat zero = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
 imgs[0] = zero;
 imgs[2] = zero;
 merge(imgs, 3, result1); //用于还原G通道的真实情况，合并结果为绿色
 imshow("result1", result1); //显示合并结果

 //输入vector参数的多通道分离与合并
 vector<Mat> imgv;
 split(HSV, imgv);
 imgv0 = imgv.at(0);
 imgv1 = imgv.at(1);
 imgv2 = imgv.at(2);
 imshow("HSV-H通道", imgv0); //显示分离后H通道的像素值
 imshow("HSV-S通道", imgv1); //显示分离后S通道的像素值
 imshow("HSV-V通道", imgv2); //显示分离后V通道的像素值
 imgv.push_back(HSV); //将vector中的图像通道数变成不统一
 merge(imgv, result2); //合并图像
  //imshow("result2", result2); /imshow最多显示4个通道，因此结果在Image Watch中查看
```



### 图像像素值统计

#### 数据类型Point:

该数据类型是用于表示图像的像素坐标，由于图像的像素坐标轴以左上角为坐标原点，水平方向为x轴，垂直方向为y轴，因此Point(x,y)对应于图像的行和列表示为Point(列数，行数)。

在OpenCV中对于2D坐标和3D坐标都设置了多种数据类型，针对2D坐标数据类型定义了整型坐标cv::Point2i(或者cv::Point)、double型坐标cv::Point2d、浮点型坐标cv::Point2f，对于3D坐标同样定义了上述的坐标数据类型，只需要将其中的数字“2”变成“3”即可。

对于坐标中x、y、z轴的具体数据，可以通过变量的x、y、z属性进行访问，例如`Point.x`可以读取坐标的x轴数据。



#### 寻找图像像素最大值与最小值`minMaxLoc()`

```c++
minMaxLoc(InputArray src,
				double * minVal,
				double * maxVal = 0,
				Point * minLoc = 0,
				Point * maxLoc = 0,
				InputArray mask = noArray()
				  )
```

- src：需要寻找最大值和最小值的图像或者矩阵，要求必须是单通道矩阵
- minVal：图像或者矩阵中的最小值。
- maxVal：图像或者矩阵中的最大值。
- minLoc：图像或者矩阵中的最小值在矩阵中的坐标。
- maxLoc：图像或者矩阵中的最大值在矩阵中的坐标。
- mask：掩模，用于设置在图像或矩阵中的指定区域寻找最值。

函数第一个参数是输入单通道矩阵，需要注意的是，该变量必须是一个单通道的矩阵数据，如果是多通道的矩阵数据，需要用`cv::Mat::reshape()`将多通道变成单通道，或者分别寻找每个通道的最值，然后再进行比较寻找到全局最值。

第二到第五个参数分别是指向最小值、最大值、最小值位置和最大值位置的指针，如果不需要寻找某一个参数，可以将该参数设置为NULL，函数最后一个参数是寻找最值得掩码矩阵，用于标记寻找上述四个值的范围，参数默认值为`noArray()`，表示寻找范围是矩阵中所有数据。

`reshape():`

```c++
reshape(int  cn,
          int  rows = 0
            )
```

- cn：转换后矩阵的通道数。
- rows：转换后矩阵的行数，如果参数为零，则转换后行数与转换前相同。



#### 代码实现 

```c++
float a[12] = { 1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 10, 0 };
Mat img = Mat(3, 4, CV_32FC1, a); //单通道矩阵
Mat imgs = Mat(2, 3, CV_32FC2, a); //多通道矩阵
double minVal, maxVal; //用于存放矩阵中的最大值和最小值
Point minIdx, maxIdx; ////用于存放矩阵中的最大值和最小值在矩阵中的位置

/*寻找单通道矩阵中的最值*/
minMaxLoc(img, &minVal, &maxVal, &minIdx, &maxIdx);
cout << "img中最大值是：" << maxVal << " " << "在矩阵中的位置:" << maxIdx << endl;
cout << "img中最小值是：" << minVal << " " << "在矩阵中的位置:" << minIdx << endl;

/*寻找多通道矩阵中的最值*/
Mat imgs_re = imgs.reshape(1, 4); //将多通道矩阵变成单通道矩阵
minMaxLoc(imgs_re, &minVal, &maxVal, &minIdx, &maxIdx);
cout << "imgs中最大值是：" << maxVal << " " << "在矩阵中的位置:" << maxIdx << endl;
cout << "imgs中最小值是：" << minVal << " " << "在矩阵中的位置:" << minIdx << endl;
```

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133582-4.png)

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133582-5.png)



#### 图像的均值`mean()`

```c++
mean(InputArray src,
 			InputArray mask = noArray()
 			  )
```

- src：待求平均值的图像矩阵。
- mask：掩模，用于标记求取哪些区域的平均值。

该函数用来求取图像矩阵的每个通道的平均值，函数的第一个参数用来输入待求平均值的图像矩阵，其通道数目可以在1到4之间。

需要注意的是，该函数的返回值是一个cv::Scalar类型的变量，函数的返回值有4位，分别表示输入图像4个通道的平均值，如果输入图像只有1个通道，那么返回值的后三位都为0，例如输入该函数一个单通道平均值为1的图像，输出的结果为[1,0,0,0]，可以通过cv::Scalar[n]查看第n个通道的平均值。

该函数的第二个参数用于控制图像求取均值的范围，在第一个参数中去除第二个参数中像素值为0的像素



#### 标准方差`meanStdDev()`

```c++
meanStdDev(InputArray src,
				 OutputArray mean,
				 OutputArray stddev,
				 InputArray mask = noArray()
				   )
```

- src：待求平均值的图像矩阵。
- mean：图像每个通道的平均值，参数为Mat类型变量。
- stddev：图像每个通道的标准方差，参数为Mat类型变量。
- mask：掩模，用于标记求取哪些区域的平均值和标准方差。

该函数的第一个参数与前面mean()函数第一个参数相同，都可以是1-4通道的图像；

不同之处在于该函数没有返回值，图像的均值和标准方差输出在函数的第二个和第三个参数中，区别于mean()函数，用于存放平均值和标准方差的是Mat类型变量，变量中的数据个数与第一个参数通道数相同；

如果输入图像只有一个通道，该函数求取的平均值和标准方差变量中只有一个数据。

#### 代码实现

![image-20231003135341124](D:\wsq\课程文件\计算机\视觉组\image-20231003135341124.png)

```c++
float a[12] = { 1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 10, 0 };
Mat img = Mat(3,4, CV_32FC1, a); //单通道矩阵
Mat imgs = Mat(2, 3, CV_32FC2, a); //多通道矩阵

cout << "/* 用meanStdDev同时求取图像的均值和标准方差 */" << endl;
Scalar myMean;
myMean = mean(imgs);
cout << "imgs均值=" << myMean << endl;
cout << "imgs第一个通道的均值=" << myMean[0] << " " 
    << "imgs第二个通道的均值=" << myMean[1] << endl << endl;

cout << "/* 用meanStdDev同时求取图像的均值和标准方差 */" << endl;
Mat myMeanMat, myStddevMat;

meanStdDev(img, myMeanMat, myStddevMat);
cout << "img均值=" << myMeanMat << " " << endl;
cout << "img标准方差=" << myStddevMat << endl << endl;
meanStdDev(imgs, myMeanMat, myStddevMat);
cout << "imgs均值=" << myMeanMat << " " << endl << endl;
cout << "imgs标准方差=" << myStddevMat << endl;
```

<img src="D:\wsq\课程文件\计算机\视觉组\640-1696682133582-6.png" alt="图片" style="zoom:50%;" />

### 图形几何形状绘制：

#### 绘制圆形`circle()`

```c++
void cv::circle(InputOutputArray img,
			        Point center,
			        int  radius,
			        const Scalar & color,
			        int  thickness = 1,
			        int  lineType = LINE_8,
			        int  shift = 0 
			        )
```

- img：需要绘制圆形的图像
- center：圆形的圆心位置坐标。
- radius：圆形的半径长度，单位为像素。
- color：圆形的颜色。
- thickness：轮廓的宽度，如果数值为负，则绘制一个实心圆。
- lineType：边界的类型，可取值为FILLED ，LINE_4 ，LINE_8 和LINE_AA（LINE_AA，消除锯齿明显，边缘光滑，但运行较慢）
- shift：中心坐标和半径数值中的小数位数。



#### 绘制直线`line()`

```c++
void cv::line(InputOutputArray img,
                 Point pt1,
                 Point pt2,
                 const Scalar & color,
                 int  thickness = 1,
                 int  lineType = LINE_8,
                 int  shift = 0 
                 )
```

- pt1：直线起始点在图像中的坐标。

- pt2：直线终点在图像中的坐标。

- color：圆形的颜色，用三通道表示。

  

#### 绘制椭圆`ellipse()`

```c++
void cv::ellipse(InputOutputArray img,
                    Point center,
                    Size axes,
                    double  angle,
                    double  startAngle,
                    double  endAngle,
                    const Scalar & color,
                    int  thickness = 1,
                    int  lineType = LINE_8,
                     int  shift = 0 
                     )
```

- center：椭圆的中心坐标。
- axes：椭圆主轴大小的一半。
- angle：椭圆旋转的角度，单位为度。
- startAngle：椭圆弧起始的角度，单位为度。
- endAngle：椭圆弧终止的角度，单位为度

另一种使用 `RotatedRect`类：

```c++
RotatedRect rrt;
rrt.center = Point(200, 200);
rrt.size = Size(100, 200);
rrt.angle = 90.0;
ellipse(bg, rrt, Scalar(0, 255, 255), 2, 8)
```

```c++
ellipse(InputOutputArray img,
			RotatedRect rrt, 
			const Scalar & color,
            int  thickness = 1,
            int  lineType = LINE_8,
            int  shift = 0 
            )
```



#### 输出椭圆的边界的像素坐标`ellipse2Poly()`

ellipse2Poly()用于输出椭圆的边界的像素坐标，但是不会在图像中绘制椭圆。

```c++
void cv::ellipse2Poly(Point center,
                           Size axes,
                           int   angle,
                           int  arcStart,
                           int   arcEnd,
                           int   delta,
                           std::vector< Point > & pts
                           )
```

- delta：后续折线顶点之间的角度，它定义了近似精度。
- pts：椭圆边缘像素坐标向量集合。



该函数与绘制椭圆需要输入的参数一致，只是不将椭圆输出到图像中，而是通过vector向量将椭圆边缘的坐标点存储起来，便于后续的再处理。对于绘制椭圆相关函数的使用我们将在本节最后的代码清单3-47中一起给出。



#### 绘制矩形`rectangle()`

```c++
void cv::rectangle(InputOutputArray img,
                       Point pt1,
                       Point pt2,
                       const Scalar & color,
                       int  thickness = 1,
                       int  lineType = LINE_8,
                       int  shift = 0 
                       )

void cv::rectangle(InputOutputArray img,
                       Rect rec,
                       const Scalar & color,
                       int  thickness = 1,
                       int  lineType = LINE_8,
                       int  shift = 0 
                       )
```

- pt1：矩形的一个顶点
- pt2：矩形中与pt1相对的顶点，即两个点在对角线上。
- rec：矩形左上角定点和长宽。



函数中与前文参数含义一致的参数不再重复介绍。在OpenCV 4中定义了两种函数原型，分别利用矩形对角线上的两个顶点的坐标或者利用左上角顶点坐标和矩形的长和宽唯一确定一个矩形。在绘制矩形时，同样可以控制边缘线的宽度绘制一个实心的矩形。

#### 数据类型Rect

该变量在OpenCV 4中表示矩形的含义，与Point、Vec3b等类型相同，都是在图像处理中常用的类型。Rect表示的是一个矩形的左上角和矩形的长和宽，该类型定义的格式为Rect(像素的x坐标，像素的y坐标，矩形的宽，矩形的高)，其中可以存放的数据类型也分别为int型（Rect2i或者Rect）、double类型（Rect2d）和float类型（Rect2f）。



#### 绘制多边形`fillPoly()`

```c++
void cv::fillPoly(InputOutputArray img,
                      const Point ** pts,
                      const int * npts,
                      int   ncontours,
                      const Scalar & color,
                      int  lineType = LINE_8,
                      int   shift = 0,
                      Point offset = Point()
                      )
```

- pts：多边形顶点数组，可以存放多个多边形的顶点坐标的数组。
- npts：每个多边形顶点数组中顶点个数。
- ncontours：绘制多边形的个数。
- offset：所有顶点的可选偏移。

函数通过依次连接多边形的顶点来实现多边形的绘制，多边形的顶点需要按照顺时针或者逆时针的顺序依次给出，通过控制边界线宽度可以实现是否绘制实心的多边形。

**pts参数是一个数组**，数组中存放的是每个多边形顶点坐标数组，**npts参数也是一个数组**，用于存放pts数组中每个元素中顶点的个数。



#### `addWeighted()`函数

是将两张相同大小，相同类型的图片融合的函数。

```c++
void cv::addWeighted( const CvArr* src1, 
                   double alpha,const CvArr* src2, 
                   double beta,double gamma, 
                   CvArr* dst 
                  );
```

参数1：src1，第一个原数组.
参数2：alpha，第一个数组元素权重

参数3：src2第二个原数组
参数4：beta，第二个数组元素权重
参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。

参数6：dst，输出图片




#### 代码实现

```c++
Rect rect;
rect.x = 100;
rect.y = 100;
rect.width = 250;
rect.height = 300;
Mat bg = Mat::zeros(image.size(), image.type());
rectangle(bg, rect, Scalar(0, 0, 255), -1, 8, 0);
circle(bg, Point(100, 100), 15, Scalar(0, 255, 0), 4, LINE_AA, 0);
line(bg, Point(100, 100), Point(350, 400), Scalar(0, 255, 0), 4, LINE_AA, 0);
RotatedRect rrt;
rrt.center = Point(200, 200);
rrt.size = Size(100, 200);
rrt.angle = 90.0;
ellipse(bg, rrt, Scalar(0, 255, 255), 2, 8);//绘制椭圆

Mat dst;
addWeighted(image, 0.7, bg, 0.3, 1, dst);
imshow("绘制演示", bg);
imshow("叠加", dst);
```



### 随机数与随机颜色

RNG可以产生3种随机数
RNG(int seed)     使用种子seed产生一个64位随机整数，默认-1
RNG::uniform( )    产生一个均匀分布的随机数
RNG::gaussian( )   产生一个高斯分布的随机数

RNG::uniform(a, b )  返回一个[a,b)范围的均匀分布的随机数，a,b的数据类型要一致，而且必须是int、float、double中的一种，默认是int。

RNG::gaussian( σ)  返回一个均值为0，标准差为σ的随机数。

​                 如果要产生均值为λ，标准差为σ的随机数，可以λ+ RNG::gaussian( σ)

![image-20231003144115676](D:\wsq\课程文件\计算机\视觉组\image-20231003144115676.png)



### 鼠标操作与响应 `setMouseCallback()`

```c++
setMouseCallback(const String & winname,
                                MouseCallback onMouse,
                                void * userdata = 0 
                                )
```

- winname：添加鼠标响应的窗口的名字
- onMouse：鼠标响应的回调函数。
- userdata：传递给回调函数的可选参数。

该函数能够为指定的图像窗口创建鼠标响应。函数第一个参数是需要创建鼠标响应的图像窗口的名字。第二个参数为鼠标响应的回调函数，该函数在鼠标状态发生改变时被调用，是一个MouseCallback类型的函数。最后一个参数是传递给回调函数的可选参数，一般情况下使用默认值0即可。

#### `MouseCallback`类型

```c++
typedef void(* cv::MouseCallback)(int  event,
                                         int  x,
                                         int  y,
                                         int  flags,
                                         void  *userdata
                                         )
```

- event：鼠标响应事件标志，参数为EVENT_*形式，具体可选参数及含义在表3-9给出。
- x：鼠标指针在图像坐标系中的x坐标
- y：鼠标指针在图像坐标系中的y坐标
- flags：鼠标响应标志，参数为EVENT_FLAG_*形式，具体可选参数及含义在表3-10给出。
- userdata：传递给回调函数的可选参数

MouseCallback类型的回调函数是一个无返回值的函数，函数名可以任意设置，有五个参数，在鼠标状态发生改变的时候被调用。函数第一个参数是鼠标响应事件标志，参数为EVENT_*形式

第二个和第三个参数分别是鼠标当前位置在图像坐标系中的x坐标和y坐标。第四个参数是鼠标响应标志，参数为EVENT_FLAG_*形式，具体可选参数

**MouseCallback类型回调函数鼠标响应事件标志可选参数及含义**

| **标志参数**        | **简记** | **含义**                           |
| ------------------- | -------- | ---------------------------------- |
| EVENT_MOUSEMOVE     | 0        | 表示鼠标指针在窗口上移动           |
| EVENT_LBUTTONDOWN   | 1        | 表示按下鼠标左键                   |
| EVENT_RBUTTONDOWN   | 2        | 表示按下鼠标右键                   |
| EVENT_MBUTTONDOWN   | 3        | 表示按下鼠标中键                   |
| EVENT_LBUTTONUP     | 4        | 表示释放鼠标左键                   |
| EVENT_RBUTTONUP     | 5        | 表示释放鼠标右键                   |
| EVENT_MBUTTONUP     | 6        | 表示释放鼠标中键                   |
| EVENT_LBUTTONDBLCLK | 7        | 表示双击鼠标左键                   |
| EVENT_RBUTTONDBLCLK | 8        | 表示双击鼠标右键                   |
| EVENT_MBUTTONDBLCLK | 9        | 表示双击鼠标中间                   |
| EVENT_MOUSEWHEEL    | 10       | 正值表示向前滚动，负值表示向后滚动 |
| EVENT_MOUSEHWHEEL   | 11       | 正值表示向左滚动，负值表示向右滚动 |

**表3-10 MouseCallback类型回调函数鼠标响应标志及含义**

| **标志参数**        | **简记** | **含义**     |
| ------------------- | -------- | ------------ |
| EVENT_FLAG_LBUTTON  | 1        | 按住左键拖拽 |
| EVENT_FLAG_RBUTTON  | 2        | 按住右键拖拽 |
| EVENT_FLAG_MBUTTON  | 4        | 按住中键拖拽 |
| EVENT_FLAG_CTRLKEY  | 8        | 按下CTRL键   |
| EVENT_FLAG_SHIFTKEY | 16       | 按下SHIFT键  |
| EVENT_FLAG_ALTKEY   | 32       | 按下ALT键    |

#### 代码实现

实现实时绘制矩形：

```c++
Point sp(-1, -1), ep(-1, -1);
Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata)
{
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN)
	{
		sp.x = x;
		sp.y = y;
		cout << "start point:" << sp.x << " " << sp.y << endl;
	}

	else if (event == EVENT_LBUTTONUP)
	{
		ep.x = x;
		ep.y = y;
		cout << "end point:" << ep.x << " " << ep.y << endl;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0)
		{
			Rect box(sp.x, sp.y, dx, dy);
            imshow("ROI区域", temp(box));//去掉红边
			rectangle(image, box, Scalar(0, 0, 255), 1, 8, 0);
			imshow("图形绘制", image);
			//为下一次做准备
			sp.x = -1;
			sp.y = -1;
		}
	}
	
	else if (event == EVENT_MOUSEMOVE)
	{
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0)
			{
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				rectangle(image, box, Scalar(0, 0, 255), 1, 8, 0);
				imshow("图形绘制", image);
			}
		}
	}
}

int main(int argc, char* argv[])
{
	Mat image = imread("newphoto.png");

	if (image.empty())
	{
		printf("could not load image...\n");
		return -1;
	}

	namedWindow("图形绘制", WINDOW_AUTOSIZE);
	setMouseCallback("图形绘制", on_draw, (void*)(&image));
	imshow("图形绘制", image);
	temp = image.clone();

	waitKey(0);
	destroyAllWindows();

	return 0;
}
```



### 图像放缩与插值

`resize() 函数`

```c++
void cv::resize (InputArray src,
                    OutputArray dst,
                    Size dsize,
                    double fx = 0,
                    double fy = 0,
                    int interpolation = INTER_LINEAR 
                    )
```

参数
src - 输入图像。
dst - 输出图像；它的大小为 dsize（当它非零时）或从 src.size()、fx 和 fy 计算的大小；dst 的类型与 src 的类型相同。
dsize - 输出图像大小；如果它等于零，则计算为：dsize = Size(round(fx*src.cols), round(fy*src.rows))。dsize 或 fx 和 fy 必须为非零。
fx - 沿水平轴的比例因子；当它等于 0 时，它被计算为(double)dsize.width/src.cols
fy - 沿垂直轴的比例因子；当它等于 0 时，它被计算为(double)dsize.height/src.rows
插值 - 插值方法，请参阅 InterpolationFlags

调整图像大小。

> dsize =（宽度，高度）
>
> 该函数resize将图像的大小src缩小到或最大到指定的大小。请注意，dst不考虑初始类型或大小。相反，大小和类型是从src、dsize、fx和派生的fy。如果要调整大小src使其适合预先创建的dst，可以调用该函数
>

要缩小图像，通常使用CV_INTER_AREA插值效果最好，而要放大图像，通常使用CV_INTER_CUBIC（慢）或CV_INTER_LINEAR（更快但看起来还可以）效果最好。

> INTER_NEAREST- 最近邻插值（最近邻插值） -
> INTER_LINEAR双线性插值（默认使用）
> INTER_AREA- 使用像素区域关系重采样。它可能是图像抽取的首选方法，因为它可以提供无莫尔条纹的结果。但是当图像被缩放时，它类似于 INTER_NEAREST 方法。
> INTER_CUBIC- 4x4 像素邻域上的双三次插值
> INTER_LANCZOS4- 8x8 像素邻域上的 Lanczos 插值
> ————————————————
> 版权声明：本文为CSDN博主「乱七八糟2333」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/JiangTao2333/article/details/122591317

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003162452661.png" alt="image-20231003162452661" style="zoom:50%;" />



### 放射变换：图像翻转与旋转

#### `flip()`翻转函数

0，上下翻转

1，左右翻转

-1，坐标点对称

![image-20231003174444700](D:\wsq\课程文件\计算机\视觉组\image-20231003174444700.png)



OpenCV 4中并没有专门用于图像旋转的函数，而是通过图像的仿射变换实现图像的旋转。实现图像的旋转首先需要确定旋转角度和旋转中心，之后确定旋转矩阵，最终通过仿射变换实现图像旋转。针对这个流程，OpenCV 4提供了getRotationMatrix2D()函数用于计算旋转矩阵和warpAffine()函数用于实现图像的仿射变换。

#### 旋转矩阵`getRotationMatrix2D()`函数

```c++
Mat cv::getRotationMatrix2D (Point2f center,
                                  double  angle,
                                  double  scale
                                  )
```

- center：图像旋转的中心位置。
- angle：图像旋转的角度，单位为度，正值为逆时针旋转。
- scale：两个轴的比例因子，可以实现旋转过程中的图像缩放，不缩放输入1。



该函数输入旋转角度和旋转中心，返回图像旋转矩阵，该返回值得数据类型为Mat类，是一个2×3的矩阵。如果我们已知图像旋转矩阵，可以自己生成旋转矩阵而不调用该函数。该函数生成的旋转矩阵与旋转角度和旋转中心的关系如式所示。

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133582-7.png)

其中：

![图片](D:\wsq\课程文件\计算机\视觉组\640-1696682133583-8.png)

确定旋转矩阵后，通过warpAffine()函数进行仿射变换，就可以实现图像的旋转:

#### 仿射变换函数`warpAffine()`

```c++
 void cv::warpAffine(InputArray src,
                         OutputArray dst,
                         InputArray M,
                         Size dsize,
                         int  flags = INTER_LINEAR,
                         int  borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar()
                        )
```

- src：输入图像。
- dst：仿射变换后输出图像，与src数据类型相同，但是尺寸与dsize相同。
- M：的变换矩阵。
- dsize：输出图像的尺寸。
- flags：插值方法标志，可选参数及含义见下表
- borderMode：像素边界外推方法的标志。
- borderValue：填充边界使用的数值，默认情况下为0。



该函数拥有多个参数，但是多数都与前面介绍的图像尺寸变换具有相同的含义。函数中第三个参数为前面求取的图像旋转矩阵，第四个参数是输出图像的尺寸。函数第五个参数是仿射变换插值方法的标志，这里相比于图像尺寸变换多增加了两个类型，可以与其他插值方法一起使用。函数第六个参数为像素边界外推方法的标志，其可以的标志和对应的方法。第七个参数是外推标志选择BORDER_CONSTANT时的定值，默认情况下为0。

**图像仿射变换中的补充插值方法**

| **标志参数**       | **简记** | **作用**                                                     |
| ------------------ | -------- | ------------------------------------------------------------ |
| WARP_FILL_OUTLIERS | 8        | 填充所有输出图像的像素，如果部分像素落在输入图像的边界外，那么他们的值设定为fillval |
| WARP_INVERSE_MAP   | 16       | 表示M为输出图像到输入图像的反变换。                          |

** 边界填充方法和对应标志**

| **标志参数**       | **简记** | **作用**                                          |
| ------------------ | -------- | ------------------------------------------------- |
| BORDER_CONSTANT    | 0        | 用特定值填充，如iiiiii\|abcdefgh\|iiiiiii         |
| BORDER_REPLICATE   | 1        | 两端复制填充，如aaaaaa\|abcdefgh\|hhhhhhh         |
| BORDER_REFLECT     | 2        | 倒叙填充，如fedcba\|abcdefgh\|hgfedcb             |
| BORDER_WRAP        | 3        | 正序填充，如cdefgh\|abcdefgh\|abcdefg             |
| BORDER_REFLECT_101 | 4        | 不包含边界值倒叙填充，如gfedcb\|abcdefgh\|gfedcba |
| BORDER_TRANSPARENT | 5        | 随机填充，uvwxyz\|abcdefgh\|ijklmno               |
| BORDER_REFLECT101  | 4        | 与BORDER_REFLECT_101相同                          |
| BORDER_DEFAULT     | 4        | 与BORDER_REFLECT_101相同                          |
| BORDER_ISOLATED    | 16       | 不关心感兴趣区域之外的部分                        |



仿射变换又称为三点变换，如果知道变换前后两张图像中三个像素点坐标的对应关系，就可以求得仿射变换中的变换矩阵，OpenCV 4提供了利用三个对应像素点来确定矩阵的函数`getAffineTransform()`：

`getAffineTransform()`

```c++
Mat cv::getAffineTransform(const Point2f src[],
                   const Point2f dst[]
                  )
```

- src[]：原图像中的三个像素坐标。
- dst[]：目标图像中的三个像素坐标。

该函数两个输入量都是存放浮点坐标的数组，在生成数组的时候像素点的输入顺序无关，但是需要保证像素点的对应关系，函数的返回值是一个变换矩阵。

#### 代码实现

```c++
Mat rotation0, rotation1, img_warp0, img_warp1;
double angle = 30; //设置图像旋转的角度
Size dst_size(img.rows, img.cols); //设置输出图像的尺寸
Point2f center(img.rows / 2.0, img.cols / 2.0); //设置图像的旋转中心
rotation0 = getRotationMatrix2D(center, angle, 1); //计算放射变换矩阵
warpAffine(img, img_warp0, rotation0, dst_size); //进行仿射变换
imshow("img_warp0", img_warp0);
//根据定义的三个点进行仿射变换
Point2f src_points[3];
Point2f dst_points[3];
src_points[0] = Point2f(0, 0); //原始图像中的三个点
src_points[1] = Point2f(0, (float)(img.cols - 1));
src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));
//放射变换后图像中的三个点
dst_points[0] = Point2f((float)(img.rows)*0.11, (float)(img.cols)*0.20);
dst_points[1] = Point2f((float)(img.rows)*0.15, (float)(img.cols)*0.70);
dst_points[2] = Point2f((float)(img.rows)*0.81, (float)(img.cols)*0.85);
rotation1 = getAffineTransform(src_points, dst_points); //根据对应点求取仿射变换矩阵
warpAffine(img, img_warp1, rotation1, dst_size); //进行仿射变换
imshow("img_warp1", img_warp1);
```

![image-20231003180218176](D:\wsq\课程文件\计算机\视觉组\image-20231003180218176.png)

就是M矩阵的第一行第一列元素结果是cos（要旋转的角度）   和M矩阵的第二行第一列的元素是sin(旋转的角度）

![image-20231003181023013](D:\wsq\课程文件\计算机\视觉组\image-20231003181023013.png)

![image-20231003204555567](D:\wsq\课程文件\计算机\视觉组\image-20231003204555567.png)



### 图像投射变换



### 归一化

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003160145916.png" alt="image-20231003160145916" style="zoom:50%;" />

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003160155199.png" alt="image-20231003160155199" style="zoom:50%;" />

L2：将所有数看成同一个高纬度向量的分量，利用模长将其单位化 也就是每个数除以模长（每个数平方相加开根号）

L1：除以和，归一化后和为1；L2：除以模，归一化后模为1；INF：除以最大值，归一化后最大值为1；MINMAX:除以最大最小差值，归一化后最大最小差值为1

**表4-1 normalize()函数归一化常用标志参数**

| **标志参数** | **简记** | **作用**             |
| ------------ | -------- | -------------------- |
| NORM_INF     | 1        | 无穷范数，向量最大值 |
| NORM_L1      | 2        | L1范数，绝对值之和   |
| NORM_L2      | 4        | L2范数，平方和之根   |
| NORM_L2SQR   | 5        | L2范数平方           |

这里是因为imshow支持浮点数数据，显示时会*255,所以会全白，imwrite就不支持浮点数数据



```c++
normalize(InputArray src,
                         InputOutputArray dst,
                         double  alpha = 1,
                         double   beta = 0,
                         int  norm_type = NORM_L2,
                         int  dtype = -1,
                         InputArray mask = noArray()
                         )
```

- src：输入数组矩阵。
- dst：输入与src相同大小的数组矩阵。
- alpha：在范围归一化的情况下，归一化到下限边界的标准值
- beta：范围归一化时的上限范围，它不用于标准规范化。
- norm_type：归一化过程中数据范数种类标志，常用可选择参数在表4-1中给出
- dtype：输出数据类型选择标志，如果为负数，则输出数据与src拥有相同的类型，否则与src具有相同的通道数和数据类型。
- mask：掩码矩阵。

该函数输入一个存放数据的矩阵，通过参数alpha设置将数据缩放到的最大范围，然后通过norm_type参数选择计算范数的种类，之后将输入矩阵中的每个数据分别除以求取的范数数值，最后得到缩放的结果。

输出结果是一个CV_32F类型的矩阵，可以将输入矩阵作为输出矩阵，或者重新定义一个新的矩阵用于存放输出结果。

该函数的第五个参数用于选择计算数据范数的种类。计算不同的范数，最后的结果也不相同，例如选择NORM_L1标志，输出结果为每个灰度值所占的比例；选择NORM_INF参数，输出结果为除以数据中最大值，将所有的数据归一化到0到1之间。



### 直方图绘制`calcHist()`

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231003210215110.png" alt="image-20231003210215110" style="zoom:50%;" />

```c++
calcHist(const Mat * images,
                   int  nimages,
                   const int * channels,
                   InputArray mask,
                   OutputArray hist,
                   int  dims,
                   const int * histSize,
                   const float ** ranges,
                   bool  uniform = true,
                    bool  accumulate = false 
                    )
```

- images：待统计直方图的图像数组，数组中所有的图像应具有相同的尺寸和数据类型，并且数据类型只能是CV_8U、CV_16U和CV_32F三种中的一种，但是不同图像的通道数可以不同。
- nimages：输入的图像数量
- channels：需要统计的通道索引数组，第一个图像的通道索引从0到images[0].channels()-1，第二个图像通道索引从images[0].channels()到images[0].channels()+ images[1].channels()-1，以此类推。
- mask：可选的操作掩码，如果是空矩阵则表示图像中所有位置的像素都计入直方图中，如果矩阵不为空，则必须与输入图像尺寸相同且数据类型为CV_8U。
- hist：输出的统计直方图结果，是一个dims维度的数组。
- dims：需要计算直方图的维度，必须是整数，并且不能大于CV_MAX_DIMS，在OpenCV 4.0和OpenCV 4.1版本中为32。
- histSize：存放每个维度直方图的数组的尺寸。
- ranges：每个图像通道中灰度值的取值范围。
- uniform：直方图是否均匀的标志符，默认状态下为均匀（true）。
- accumulate：是否累积统计直方图的标志，如果累积（true），则统计新图像的直方图时之前图像的统计结果不会被清除，该同能主要用于统计多个图像整体的直方图。



该函数用于统计图像中每个灰度值像素的个数，例如统计一张CV_8UC1的图像，需要统计灰度值从0到255中每一个灰度值在图像中的像素个数，如果某个灰度值在图像中没有，那么该灰度值的统计结果就是0。

由于该函数具有较多的参数，并且每个参数都较为复杂，因此建议在使用该函数时只统计单通道图像的灰度值分布，对于多通道图像可以将图像每个通道分离后再进行统计。

#### 代码实现：

```c++
//三通道分离
	vector<Mat> bgr_plane;
	split(image, bgr_plane);
	//定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	Mat b_hist, g_hist, r_hist;
	//计算Blue, Green, Red通道的直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	//显示直方图
	int hist_w = 512;//设置直方图图像的宽度为512像素
	int hist_h = 400;//设置直方图图像的高度为400像素
	int bin_w = cvRound((double)hist_w / bins[0]);//计算每个直方图条的宽度，以便在图像中绘制
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	//归一化直方图数据
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//绘制直方图曲线
	for (int i = 1; i < bins[0]; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}

	//显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
```

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231005000720715.png" alt="image-20231005000720715" style="zoom:50%;" />

另一种方式：

```c++
Mat gray;
cvtColor(img, gray, COLOR_BGR2GRAY);
//设置提取直方图的相关变量
Mat hist; //用于存放直方图计算结果
const int channels[1] = { 0 }; //通道索引
float inRanges[2] = { 0,255 };
const float* ranges[1] = { inRanges }; //像素灰度值范围
const int bins[1] = { 256 }; //直方图的维度，其实就是像素灰度值的最大值
calcHist(&img, 1, channels, Mat(), hist, 1, bins, ranges); //计算图像直方图
//准备绘制直方图
int hist_w = 512;
int hist_h = 400;
int width = 2;
Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
for (int i = 1; i <= hist.rows; i++)
{
  rectangle(histImage, Point(width*(i - 1), hist_h - 1),
    Point(width*i - 1, hist_h - cvRound(hist.at<float>(i - 1) / 20)),
    Scalar(255, 255, 255), -1);
}
namedWindow("histImage", WINDOW_AUTOSIZE);
imshow("histImage", histImage);
imshow("gray", gray);
```

<img src="D:\wsq\课程文件\计算机\视觉组\640-1696682133583-9.jpeg" alt="图片" style="zoom:50%;" />





#### 二维直方图



所用函数与一维直方图的一样。

代码实现：

```c++
Mat hsvImage;
 
        //因为要计算H-S的直方图，所以需要先进行颜色空间的转换
        cvtColor(srcImage, hsvImage, CV_BGR2HSV);
 
 
        //计算H-S二维直方图的参数配置
        int channels[] = { 0, 1 };
        Mat dstHist;
        int histSize[] = { 30, 32 };
        float HRanges[] = { 0, 181 };
        float SRanges[] = { 0, 256 };
        const float *ranges[] = { HRanges, SRanges };
 
        //参数配置好后,就可以调用calcHis函数计算直方图数据了
        calcHist(&hsvImage, 1, channels, Mat(), dstHist, 2, histSize, ranges, true, false);
 
        //calcHist函数调用结束后，dstHist变量中将储存直方图数据 
 
        ///接下来绘制直方图
        //首先先创建一个黑底的图像，为了可以显示彩色，所以该绘制图像是一个8位的3通道图像  
        Mat drawImage = Mat::zeros(Size(300, 320), CV_8UC3);
 
        //因为数据矩阵中的某个值的总个数可能会超出所定义的图像的尺寸，所以要对个数进行归一化处理  
        //先用 minMaxLoc函数来得到计算直方图中的最大数据,这个函数的使用方法大家一看函数原型便知
        double g_dHistMaxValue;
        minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
 
        //遍历直方图数据,对数据进行归一化和绘图处理 
        for (int i = 0; i < 30; i++)
        {
                for (int j = 0; j < 32; j++)
                {
                        int value = cvRound(dstHist.at<float>(i, j) * 255/ g_dHistMaxValue);
 
                        rectangle(drawImage, Point(10 * i, j * 10), 
                                  Point((i + 1) * 10 - 1, (j + 1) * 10 - 1), 
                                  Scalar(0,0,value), 
                                  /*或者这里用Scalar::all(intensity),再使用
                                  applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET); 将灰度图变成彩色图，									颜色空间转换函数*/
                                  -1);
                    //这两句代码把hist里的数据归一化到0~255。
                    //这两句代码把二维直方图进行了按比例放大绘制，本来hist的大小是30×32的，但在绘制的时候我们把其放大到了					300×320，这是为了便于我们观察。
                    //我们是用三通道的彩色图绘制二维直方图的，这样做也是便于我们肉眼观察，每一块矩形我们都用不同程度的红色来绘					  制，其红色的程度值就是hist里对应点进行归一化后的值。
                }
        }
 		//将灰度图变成彩色图，颜色空间转换函数
		applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
        imshow("【H-S二维联合直方图】", drawImage);
```

结果：

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231007144113751.png" alt="image-20231007144113751" style="zoom:33%;" />

图形颜色这里，使用`Scalar::all(intensity)`,再使用`applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);` 将灰度图变成彩色图，颜色空间转换函数,结果如图，更明显一点

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231007143750341.png" alt="image-20231007143750341" style="zoom:33%;" />



### 直方图均衡化`equalizeHist()`

如果一个图像的直方图都集中在一个区域，则整体图像的对比度比较小，不便于图像中纹理的识别。例如相邻的两个像素灰度值如果分别是120和121，仅凭肉眼是如法区别出来的。同时，如果图像中所有的像素灰度值都集中在100到150之间，则整个图像想会给人一种模糊的感觉，看不清图中的内容。如果通过映射关系，将图像中灰度值的范围扩大，增加原来两个灰度值之间的差值，就可以提高图像的对比度，进而将图像中的纹理突出显现出来，这个过程称为图像直方图均衡化。

在OpenCV 4中提供了`equalizeHist()函数`用于将图像的直方图均衡化:



```c++
void cv::equalizeHist(InputArray src,
                           OutputArray dst
                           )
```

- src：需要直方图均衡化的CV_8UC1图像。
- dst：直方图均衡化后的输出图像，与src具有相同尺寸和数据类型。

该函数形式比较简单，但是需要注意该函数只能对单通道的灰度图进行直方图均衡化。

![image-20231003222647882](D:\wsq\课程文件\计算机\视觉组\image-20231003222647882.png)

### 卷积操作

#### 高斯模糊`GaussianBlur()`

高斯模糊，也叫高斯平滑，英文为：Gaussian Blur，是图像处理中常用的一种技术，主要用来降低图像的噪声和减少图像的细节。高斯模糊在许多图像处理软件中也得到了广泛的应用。

模糊在图像中的意思可理解为：中心像素的像素值为由周围像素的像素值的和的平均值。如图：

第一幅图为原始图像，其中心像素的像素值为2，第二幅图为中心像素进行模糊后的图像，其像素值为周围像素值的和的平均值。
图像模糊在数值上，这是一种”平滑化”。在图形上，就相当于产生”模糊”效果，”中心点”失去细节。高斯模糊会减少图像的高频信息，因此是一个低通滤波器。


```c++
void cv::GaussianBlur(InputArray  src,  
							OutputArray  dst,                   
							Size  ksize,                  
							double  sigmaX,                
							double  sigmaY = 0,             
							int  borderType = BORDER_DEFAULT           
                            )
```

- src：待高斯滤波图像，图像可以具有任意的通道数目，但是数据类型必须为CV_8U，CV_16U，CV_16S，CV_32F或CV_64F。
- dst：输出图像，与输入图像src具有相同的尺寸、通道数和数据类型。
- ksize：高斯滤波器的尺寸，滤波器可以不为正方形，但是必须是正奇数。如果尺寸为0，则由标准偏差计算尺寸。
- sigmaX：X方向的高斯滤波器标准偏差。
- sigmaY：Y方向的高斯滤波器标准偏差; 如果输入量为0，则将其设置为等于sigmaX，如果两个轴的标准差均为0，则根据输入的高斯滤波器尺寸计算标准偏差。
- borderType：像素外推法选择标志，默认参数为BORDER_DEFAULT，表示不包含边界值倒序填充。

该函数能够根据输入参数自动生成高斯滤波器，实现对图像的高斯滤波。该函数第三个参数是高斯滤波器的尺寸，该函数除了必须是正奇数以外，还允许输入尺寸为0，当输入的尺寸为0时，会根据输入的标准偏差计算滤波器的尺寸。函数第四个和第五个参数为X方向和Y方向的标准偏差，当Y方向参数为0时表示Y方向的标准偏差与X方向相同，当两个参数都为0时，则根据输入的滤波器尺寸计算两个方向的标准偏差数值。但是为了能够使计算结果符合自己的预期，建议将第三个参数、第四个参数和第五个参数都明确的给出。

#### 高斯双边模糊

<img src="D:\wsq\课程文件\计算机\视觉组\image-20231007144403425.png" alt="image-20231007144403425" style="zoom: 50%;" />



![image-20231005001948624](D:\wsq\课程文件\计算机\视觉组\image-20231005001948624.png)

![image-20231005002007624](D:\wsq\课程文件\计算机\视觉组\image-20231005002007624.png)

