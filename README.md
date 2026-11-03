# RustCnOcr

- 一份来自python cnocr的Rust实现，官方文档->[CnOcr文档](https://cnocr.readthedocs.io/zh/latest/)
- 全部依赖都来自 crates.io，不需要再安装任何本地 C/C++ 库
- 识别流程参考了同作者的 C++ 实现 CnocrCpp：多模型注册表、按 token 读取字典、
  依据模型元数据自动组装输入输出、投影切分文字行，以及 PP-OCRv3 文本行检测（`cnstd` 模块）

## 依赖

- `ort`：ONNX Runtime 的 Rust 绑定，构建时自动下载对应平台的 onnxruntime 预编译库（`download-binaries`），
  不再需要手动下载 C++ 版 onnxruntime、也不需要改 build.rs
  - 版本暂用 `=2.0.0-rc.13` 固定，等 ort 2.0.0 正式版发布后可以放宽；rc.15 起要求 rustc 1.92
  - 计算后端为 CPU；原 C++ 实现里的 TensorRT 加速没有迁移，需要 GPU 时给 `ort` 打开 `cuda` / `tensorrt` 特性
  - 内网/离线环境可以设 `ORT_LIB_LOCATION` 指向已有的 onnxruntime，或改用 `load-dynamic` 特性
- `image`：图片解码与缩放，纯 Rust 实现，替换了原来的 `opencv` 依赖
- `ureq` + `flate2`：模型自动下载与压缩包解包。这两个包本来就被 `ort` 的 `download-binaries`
  拉进依赖树，显式沿用 `ort` 的 `native-tls`（Windows 走系统 schannel），不会引入需要 C 编译器的
  `rustls`/`ring`
- 检测后处理里的几何计算也是纯 Rust 手写（`geometry` 模块）：连通域标注、点集最小外接矩形、
  多边形外扩、单应矩阵求解与双线性透视采样，分别对应 OpenCV 的 `findContours`、`minAreaRect`、
  pyclipper offset、`getPerspectiveTransform` + `warpPerspective`

## 模型与字典

识别/检测模型首次使用时会自动从
[hf-mirror 的 breezedeus/cnstd-cnocr-models](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)
下载 `.onnx.zip` 并解包到缓存目录；已经存在的模型不会重复下载。查找顺序为
`modle/` → 当前目录 → `<CNOCR_HOME>/models`（未设 `CNOCR_HOME` 时，Windows 用 `%APPDATA%\cnocr`，
其他平台用 `~/.cnocr`）。

镜像地址由环境变量控制：

| 变量 | 作用 |
| --- | --- |
| `CNOCR_MODEL_ENDPOINT` | 非空时只用这一个地址，适合内网自建镜像 |
| `HF_ENDPOINT` | 排在默认列表最前，后面仍会回落到 `hf-mirror.com`、`huggingface.co` |
| `CNOCR_HOME` | 模型缓存根目录（实际写入 `<CNOCR_HOME>/models`） |

字典文件不在上面这个仓库里，仍需手动放到 `dict/` 或当前目录。关闭自动下载：
`Ocr::builder().auto_download(false)`，命令行加 `--no-download`；此时缺文件会直接报错并列出查找过的位置。

内置可选的识别模型：

| 模型名 | 通道 | 输入高度 | 字典 | 说明 |
| --- | --- | --- | --- | --- |
| `cnocr136fc` | 1 | 32 | `label_cn.txt` | 早期中文识别模型（cnocr 2.2），默认 |
| `densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 通用中文识别模型（cnocr 2.3） |
| `scene-densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 场景文字识别模型 |
| `doc-densenet_lite_136-gru` | 1 | 32 | `label_cn_2.3.txt` | 文档识别模型 |
| `number-densenet_lite_136-fc` | 1 | 32 | `label_number.txt` | 数字识别模型 |
| `en_number_mobile_v2.0` | 3 | 32 | `en_dict.txt` | 英文数字识别模型（PaddleOCR） |
| `chinese_cht_PP-OCRv3` | 3 | 32 | `chinese_cht_dict.txt` | 繁体中文识别模型 |
| `ch_PP-OCRv3` | 3 | 48 | `ppocr_keys_v1.txt` | 简体中英文识别模型 |
| `ch_ppocr_mobile_v2.0` | 3 | 32 | `ppocr_keys_v1.txt` | 简体中英文识别模型（mobile v2.0） |

文本行检测模型（不需要字典）：

| 模型文件 | 输入长边 | 说明 |
| --- | --- | --- |
| `ch_PP-OCRv3_det_infer.onnx` | 768 | DBNet 文本行检测，适合倾斜、场景图 |

```rust
use cnocr_rs::ocr::Ocr;

// 默认模型 + 投影切分
let ocr = Ocr::new();

// 指定模型，并改用 PP-OCRv3 检测模型切分文本行
let ocr = Ocr::builder().model_name("ch_PP-OCRv3").detection(true).build();

// 离线场景：只用本地已有的模型，缺失就报错而不是下载
let ocr = Ocr::builder().auto_download(false).build();
```

> 仓库自带 `label_cn.txt`，首次运行会自动下载 `cnocr136fc.onnx`，开箱即用；换模型只需要放好对应字典，
> 模型文件交给自动下载。
> 投影切分只处理白底黑字，深色背景会先自动反色；检测切分同样会先做反色归一。

## 与旧版本的接口差异

去掉 `opencv` 后，接口类型不再是 `Mat`：

| 旧接口 | 新接口 |
| --- | --- |
| `from_mat(Mat)` | `from_image(&DynamicImage)` |
| `from_path(String)` | `from_path(&str)` |
| `ocr_for_single_lines(Vec<Mat>)` | `ocr_for_single_lines(&[GrayImage])` |
| `line_split(Mat) -> Vec<Mat>` | `line_split(&GrayImage) -> Vec<GrayImage>` |
| `ctc_best(Vec<usize>)` | `ctc_best(&[usize])` |
| `Ocr::new()` | 仍是 `Ocr::new()`，构造参数改走 `Ocr::builder()`（`model` / `model_name` / `detection` / `detection_options` / `auto_download`） |

`label_cn.txt` 换成了与 Python cnocr / CnocrCpp 一致的版本：占位标签写作 `<unused1>`…，
下标与字符严格对齐，`<space>` 会被读成空格。旧版把占位行存成了空格，识别结果里会夹进多余空格。

## 待解决问题

- 检测切分的文本框比投影切分更紧，行尾字符（尤其标点）偶尔会被裁掉；
  用 `Ocr::builder().detection_options(..)` 调大 `unclip_ratio` 可以补回行尾，但框太大会带进相邻文字并拉低置信度
- 字典文件没有托管地址，仍需手动放置；`ch_PP-OCRv3` 等模型的字典要自己去 Python cnocr / PaddleOCR 取

## 使用演示

```rust
use cnocr_rs::ocr::{Ocr, OcrTraitConst};
fn main() {
    let ocr = Ocr::new();
    let vs = ocr.from_path("cnocr.png");
    for i in vs {
        println!("识别到:{:?}", i);
    }
}
```
```sh
识别到:("cnocr自V2.1.2之后，可直接使用的模型包含两类：1）cnocr自己训练的模型，通常会包含PyTorch和", 0.5027224)
识别到:("ONNX版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 cnocr中。", 0.71831757)
```

命令行示例支持指定图片、模型，以及用 `-d` 开启文本行检测：

```sh
cargo run --release -- cnocr.png cnocr136fc          # 投影切分
cargo run --release -- cnocr.png cnocr136fc -d       # PP-OCRv3 检测切分
cargo run --release -- cnocr.png cnocr136fc --no-download  # 只用本地模型，缺失即报错
```

单独使用检测器（只要文本框，不做识别）：

```rust
use cnocr_rs::cnstd::{crop_line, CnStd, DetectOptions};

let gray = image::open("cnocr.png").unwrap().to_luma8();
for line in CnStd::new().detect(&gray, &DetectOptions::default()) {
    let crop = crop_line(&gray, &line).unwrap();
    println!("{:?} 得分={} 裁剪={}x{}", line.points, line.score, crop.width(), crop.height());
}
```

回归测试：`cargo test`（需要仓库根目录下的字典与示例图片，模型缺失时会自动下载）。
真实联网下载的用例默认跳过，用 `cargo test --release -- --ignored` 手动跑。
