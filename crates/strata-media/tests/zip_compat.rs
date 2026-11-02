//! candle reads weights through zip 8's API, which vendor/zip keeps on top of zip 9 (vendor/zip/AM2STRATA.md): every path
//! that calls it, on a real archive.

use candle_core::npy::NpzTensors;
use candle_core::{pickle, Device, Tensor};

#[test]
fn npz_and_pth_read_through_the_vendored_zip() {
    let path = std::env::temp_dir().join(format!("am2-zip-{}.npz", std::process::id()));
    let t = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
    Tensor::write_npz(&[("weight", t.clone())], &path).unwrap();

    // name() through by_index, with the suffix taken off
    let all = Tensor::read_npz(&path).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, "weight");
    assert_eq!(all[0].1.to_vec2::<f32>().unwrap(), t.to_vec2::<f32>().unwrap());
    // by_name, and name() in NpzTensors' index
    assert_eq!(Tensor::read_npz_by_name(&path, &["weight"]).unwrap().len(), 1);
    let z = NpzTensors::new(&path).unwrap();
    assert_eq!(z.names(), vec!["weight"]);
    assert!(z.get("weight").unwrap().is_some());
    // file_names(): an archive with no data.pkl holds no tensors, and reading it is not an error
    assert!(pickle::read_pth_tensor_info(&path, false, None).unwrap().is_empty());
    let _ = std::fs::remove_file(&path);
}
