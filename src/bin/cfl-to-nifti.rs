use std::path::PathBuf;
use clap::Parser;
use cfl::{dump_magnitude,dump_phase,dump_real,dump_imaginary};

#[derive(clap::Parser,Debug)]
struct Args {
    input_cfl_base: PathBuf,
    output_nifti_base: PathBuf,
    /// write the phase
    #[clap(short='p')]
    write_phase: bool,
    /// write the real part
    #[clap(short='r')]
    write_real: bool,
    /// write the imaginary part
    #[clap(short='i')]
    write_imaginary: bool,
}

fn main() {
    let args = Args::parse();
    let cfl = cfl::to_array(args.input_cfl_base,true).expect("failed to load cfl");
    if args.write_phase {
        dump_phase(args.output_nifti_base,&cfl);
    }else if args.write_real {
        dump_real(args.output_nifti_base,&cfl);
    }else if args.write_imaginary {
        dump_imaginary(args.output_nifti_base,&cfl);
    }else {
        dump_magnitude(args.output_nifti_base,&cfl);
    }
}