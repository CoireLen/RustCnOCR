#![allow(dead_code, unused_imports)]
struct connxmodle{
    onnxmodle:std::ffi::c_void,
}
struct connxret{
    output_lengths:*const std::ffi::c_longlong,
    logits:*const std::ffi::c_float,
}
extern "C"{
    fn connxmodleInit()->connxmodle;
    fn connxmodleRun(modle:*const connxmodle ,input_lengths:std::ffi::c_longlong,x_length:std::ffi::c_longlong,x:*const u8)->connxret;
}
pub mod Ocr{
    use opencv::prelude::MatTraitConstManual;

    use crate::{connxmodle, connxmodleInit, connxmodleRun};
    struct onnxmodle{
        

    }
    //impl onnxmodle{
    //    fn new()->onnxmodle{
    //        let env=onnxruntime::environment::Environment::builder()
    //        .with_name("test").with_log_level(onnxruntime::LoggingLevel::Error)
    //        .build().unwrap();
    //        let mut session=env.new_session_builder().unwrap()
    //        .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic).unwrap()
    //        .with_number_threads(1).unwrap()
    //        .with_model_from_file("").unwrap();
    //        
    //    }
    //}
    pub struct ocr{
        ctc_path:String,
        ctc_data:Vec<String>,
        modle:connxmodle,
    }
    impl ocr {
        pub fn new()->ocr{
            let m:connxmodle;
            unsafe{
                m=connxmodleInit();
            }
            ocr{ctc_path:"label_cn.txt".to_string(),
                ctc_data:Vec::<String>::new(),
                modle:m
            }
        }
    }
    pub trait OcrTraitConst {
        fn from_mat(&self,imgdata:opencv::core::Mat)->Vec<String>;
        fn from_path(&self,path:String)->Vec<String>;
        fn ocr_for_single_lines(&self,inimgs:Vec<opencv::core::Mat>)->Vec<String>;
        fn line_split(&self,inimg:opencv::core::Mat)->Vec<opencv::core::Mat>;
        fn ctc_best(data:Vec<u16>);
    }
    impl OcrTraitConst for ocr{
        fn from_mat(&self,imgdata:opencv::core::Mat)->Vec<String>{
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            let mut outimg=opencv::core::Mat::default();
            if let Err(_)=opencv::imgproc::cvt_color(&imgdata, &mut outimg,opencv::imgproc::COLOR_RGB2GRAY,0){
                assert!(false,"From_mat.cvt_color(imgdata,outimg) faild");
            }
            let mut res:Vec<String>=Vec::new();
            let imgcol=outimg.cols();
            let imgrow=outimg.rows();
            if std::cmp::min(imgcol,imgrow)<2{
                return res;
            }
            if opencv::core::sum_elems(&outimg.col(0).unwrap()).unwrap()[0]>145.0{
                let o=opencv::core::Mat::ones(imgrow, imgcol, opencv::core::CV_8UC1).unwrap();
                if let opencv::core::MatExprResult::Ok(i)=(255f64*o)-&outimg{
                    outimg=i.to_mat().unwrap();
                }
            }
            let imgs=self.line_split(outimg);
            res=self.ocr_for_single_lines(imgs);
            res
        }
        fn from_path(&self,path:String)->Vec<String>{
            use opencv::prelude::MatTraitConst;
            let inimg =opencv::imgcodecs::imread(path.as_str(), opencv::imgcodecs::IMREAD_COLOR).unwrap();
            return self.from_mat(inimg);
        }
        fn  ocr_for_single_lines(&self,inimgs:Vec<opencv::core::Mat>)->Vec<String>{
            let mut res:Vec<String>=Vec::new();
            if inimgs.len()==0{
                return res;
            }
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            for img in inimgs{
                let imgcol=img.cols();
                let imgrow=img.rows();
                let mut imgmat=opencv::core::Mat::default();
                let ratio=32.0/imgrow as f64;
                let sz=opencv::core::Size::new(ratio as i32, 32);
                opencv::imgproc::resize(&img,&mut imgmat,opencv::core::Size::default(), ratio , ratio,  0).unwrap();
                println!("copyto modle imgsize({},{})",imgmat.cols(),imgmat.rows());//测试将进入模型的数据是否高度为32
                unsafe{
                    let sz=imgmat.size().unwrap();
                    let ret=connxmodleRun(&self.modle, sz.width as i64,(sz.height*sz.width) as i64, imgmat.data());
                    let length=*ret.output_lengths;
                    println!("connx:{}",length);
                }
                
                //接下来用softmax处理信息

            }
           
            res
        }
        /// # line_split
        /// > 将黑白图片每行找出最大值，列出竖直列，其中较大部分为文字。将其切分成 图片 每张图片仅只能有一行文字。
        ///
        fn line_split(&self,inimg:opencv::core::Mat)->Vec<opencv::core::Mat>{

            let mut list:Vec<opencv::core::Mat>=Vec::new();
            use opencv::prelude::MatTraitConst;
            use opencv::core::prelude::*;
            use opencv::prelude::*;
            use opencv::core::*;
            
            let imgcol=inimg.cols();
            let imgrow=inimg.rows();
            println!("img({},{})",imgcol,imgrow);
            let mut tmp=opencv::core::Mat::default();
            opencv::core::reduce_arg_max(&inimg, &mut tmp, 1, false).unwrap();
            let tmpdata=tmp.data_bytes().unwrap();
            let mut splitlinedata:Vec<u8>=Vec::new();
            for i in 0..(tmpdata.len()/4){
                splitlinedata.push(tmpdata[i]);
            }
            let mut lineforchar=0;
            let mut lineforcharstart=0;
            for i in 0..splitlinedata.len(){
                if splitlinedata[i]<100{
                    if lineforchar+1==i{
                        lineforchar+=1;
                    }
                    else{
                        if lineforchar-lineforcharstart>7{
                            let mut start=lineforcharstart;
                            let mut end=lineforchar;
                            if (start>0){
                                start-=1;
                            }
                            let mut tmpvec:opencv::core::Vector<Range>=opencv::core::Vector::new();
                            tmpvec.push(Range::new(start as i32, end as i32).unwrap());
                            tmpvec.push(Range::all().unwrap());
                            let mut tmpimg=opencv::core::Mat::default();
                            tmpimg=opencv::core::Mat::ranges(&inimg, &tmpvec).unwrap();
                            list.push(tmpimg);
                        
                        }
                        lineforcharstart=i;
                        lineforchar=i;
                    }
                }
            }
            if lineforchar-lineforcharstart>5{
                let mut start=lineforcharstart;
                let mut end= lineforchar;
                if (start>0){
                    start-=1;
                }
                if end<(imgcol-1)as usize{
                    end+=1;
                }
                let mut tmpvec:opencv::core::Vector<Range>=opencv::core::Vector::new();
                tmpvec.push(Range::new(start as i32, end as i32).unwrap());
                tmpvec.push(Range::all().unwrap());
                let mut tmpimg=opencv::core::Mat::default();
                tmpimg=opencv::core::Mat::ranges(&inimg, &tmpvec).unwrap();
                list.push(tmpimg);
            }
            //for i in &list{
            //    println!("({},{})",i.cols(),i.rows());
            //    opencv::highgui::imshow("", i);
            //    opencv::highgui::wait_key(0);
            //}
            list
        }
        fn ctc_best(data:Vec<u16>){

        }
    }
}