use cnocr_rs::ocr::{Ocr, OcrTraitConst};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // 用法: cnocr-rs [图片路径] [模型名]
    let path = args.get(0).map(String::as_str).unwrap_or("cnocr.png");
    let t1 = Instant::now();
    let ocr = match args.get(1) {
        Some(name) => Ocr::with_model_name(name),
        None => Ocr::new(),
    };
    println!("模型:{} 载入用时:{}ms", ocr.model().name, t1.elapsed().as_millis());
    let t2 = Instant::now();
    for i in ocr.from_path(path) {
        println!("识别到:{:?}", i);
    }
    println!("识别用时:{}ms", t2.elapsed().as_millis());
    println!("总用时:{}ms", t1.elapsed().as_millis());
}
