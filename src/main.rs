use anyhow::{bail, Result};
use tch::vision::{
    imagenet,
    resnet,
};
use tch::{Device, nn::ModuleT};

fn main() -> Result<()> {
    let image_file = "tiger_maybe.jpg";
    let image = imagenet::load_image_and_resize224(image_file)?;
    let model_file = "resnet18.ot";
    let weights = std::path::Path::new(model_file);
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net: Box<dyn ModuleT> = Box::new(resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT));
    vs.load(weights)?;

    let output = net.forward_t(&image.unsqueeze(0), false).softmax(-1, tch::Kind::Float); 

    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability);
    }
    Ok(())
}
