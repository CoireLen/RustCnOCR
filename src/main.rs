use cnocr_rs::ocr::{Ocr,OcrTraitConst};
fn main() {
    let ocr=Ocr::new();
    let vs=ocr.from_path("cnocr.png".to_string());
    for i in vs{
        println!("识别到:{:?}",i);
    }
}
