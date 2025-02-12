use std::path::{Path, PathBuf};
use clap::Parser;
use cfl::{dump_magnitude, dump_phase, dump_real, dump_imaginary, read_nifti_to_cfl};

#[derive(clap::Parser,Debug)]
struct Args {
    input_nifti: PathBuf,
    output_cfl_base: PathBuf,
    /// read imaginary part
    #[clap(short='i')]
    read_imaginary: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let cfl = read_nifti_to_cfl(&args.input_nifti,args.read_imaginary.as_ref());
    cfl::from_array(args.output_cfl_base,&cfl).expect("failed to write cfl data");
}