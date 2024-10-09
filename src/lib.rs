use memmap::Mmap;
use memmap::MmapMut;
use memmap::MmapOptions;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::IntoParallelRefMutIterator;
use ndarray::parallel::prelude::ParallelIterator;
use ndarray::ArrayD;
use ndarray::ShapeBuilder;
use num_complex::Complex32;
use regex::Regex;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::BufWriter;
use std::io::Read;
use std::io::Seek;
use std::io::Write;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;
use std::slice;

pub use num_complex;
pub use ndarray;
pub use ndarray_stats;

#[derive(Debug)]
pub enum CflError {
    DimensionsNotFound,
    TooManyDims(usize),
    IO(io::Error),
    Mmap,
    InvalidMemOrder,
    CflNotFound(PathBuf),
    HdrNotFound(PathBuf),
    MmapFlush,
    BufWriter,
    BufWriterFlush,
}

pub struct CflReader {
    mmap:Mmap,
    n_elements:usize,
    dimensions:Vec<usize>,
}

impl CflReader {
    pub fn new(file_path:impl AsRef<Path>) -> Result<Self,CflError> {
        let dimensions = get_dims(&file_path)?;
        let f = File::open(file_path.as_ref().with_extension("cfl")).map_err(|e|CflError::IO(e))?;
        let mmap = unsafe { Mmap::map(&f).map_err(|_|CflError::Mmap)? };
        let n_elements = mmap.len() / size_of::<Complex32>();
        Ok(
            Self {
                mmap,
                n_elements,
                dimensions
            }
        )
    }

    /// random parallel file reads from index values into buffer
    pub fn read_into(&self,indices:&[usize],buffer:&mut [Complex32]) -> Result<(),CflError> {
        buffer.par_iter_mut().zip(indices.par_iter()).for_each(|(dst,src_idx)|{
            *dst = self.read(*src_idx).unwrap()
        });
        Ok(())
    }

    pub fn read_slice(&self,idx:usize,buffer:&mut [Complex32]) -> Result<(),CflError> {
        if idx + buffer.len() > self.n_elements {
            Err(CflError::Mmap)?
        }

        let start = idx*size_of::<Complex32>();

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mmap[start..(start+size_of::<Complex32>()*buffer.len())].as_ptr(),
                buffer.as_mut_ptr() as *mut u8,
                buffer.len() * size_of::<Complex32>(),
            );
        }
        Ok(())

    }

    pub fn read(&self,idx:usize) -> Result<Complex32,CflError> {
        let mut buff = [Complex32::ZERO];
        self.read_slice(idx, &mut buff)?;
        Ok(buff[0])
    }

}

// pub struct CflBufWriter {
//     writer: BufWriter<File>,
//     n_elements:usize,
// }

// impl CflBufWriter {
//     pub fn new(cfl_base:impl AsRef<Path>, dimensions:&[usize]) -> Result<Self,CflError> {
//         let f = OpenOptions::new()
//         .read(true)
//         .write(true)
//         .create(true)
//         .truncate(true)
//         .open(cfl_base.as_ref().with_extension("cfl"))
//         .unwrap();

//         let n_elements = dimensions.iter().product::<usize>();
//         f.set_len(
//             (n_elements * size_of::<Complex32>()) as u64
//         ).map_err(|e|CflError::IO(e))?;
        
//         write_header(cfl_base, dimensions)?;

//         let writer = BufWriter::new(f);

//         Ok(Self {
//             writer,
//             n_elements,
//         })
//     }

//     pub fn open(cfl_base:impl AsRef<Path>) -> Result<Self,CflError> {
//         let dimensions = get_dims(&cfl_base)?;
//         let f = OpenOptions::new()
//         .read(true)
//         .write(true)
//         .open(cfl_base.as_ref().with_extension("cfl"))
//         .unwrap();
//         let n_elements = dimensions.iter().product::<usize>();

//         let writer = BufWriter::new(f);
//         Ok(Self {
//             writer,
//             n_elements,
//         })
//     }

//     pub fn write_slice(&mut self,dest_idx:usize,src:&[Complex32]) -> Result<(),CflError> {
//         if dest_idx + src.len() > self.n_elements {
//             Err(CflError::Mmap)?
//         }
//         let start = dest_idx*size_of::<Complex32>();
//         self.writer.seek(io::SeekFrom::Start(start as u64)).unwrap();
//         let ptr = src.as_ptr() as *const u8;
//         // Calculate the byte length of the slice
//         let len = src.len() * size_of::<Complex32>();
//         // Create a byte slice from the raw pointer and length
//         let byte_slice = unsafe { slice::from_raw_parts(ptr, len) };
//         self.writer.write_all(byte_slice).map_err(|_|CflError::BufWriter)?;
//         self.writer.flush().map_err(|_|CflError::BufWriterFlush)?;
//         Ok(())
//     }

//     pub fn write(&mut self,idx:usize,value:Complex32) -> Result<(),CflError> {
//         self.write_slice(idx, &[value])?;
//         Ok(())
//     }

// }



pub struct CflWriter {
    mmap:MmapMut,
    //dimensions:Vec<usize>,
    n_elements:usize,
}

impl CflWriter {
    pub fn new(cfl_base:impl AsRef<Path>, dimensions:&[usize]) -> Result<Self,CflError> {
        let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(cfl_base.as_ref().with_extension("cfl"))
        .unwrap();

        let n_elements = dimensions.iter().product::<usize>();
        f.set_len(
            (n_elements * size_of::<Complex32>()) as u64
        ).map_err(|e|CflError::IO(e))?;

        write_header(cfl_base, dimensions)?;

        let mmap = unsafe {
            MmapMut::map_mut(&f).map_err(|_|CflError::Mmap)?
        };

        Ok(Self {
            mmap,
            //dimensions: dimensions.to_owned(),
            n_elements,
        })

    }

    pub fn open(cfl_base:impl AsRef<Path>) -> Result<Self,CflError> {
        let dimensions = get_dims(&cfl_base)?;
        let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(cfl_base.as_ref().with_extension("cfl"))
        .unwrap();
        let n_elements = dimensions.iter().product::<usize>();
        let mmap = unsafe {
            MmapMut::map_mut(&f).map_err(|_|CflError::Mmap)?
        };
        Ok(Self {
            mmap,
            //dimensions: dimensions.to_owned(),
            n_elements,
        })
    }

    pub fn write_from(&mut self,dst_indices:&[usize],src:&[Complex32]) -> Result<(),CflError> {
        src.iter().zip(dst_indices.iter()).for_each(|(value,dst_idx)|{
            self.write(*dst_idx,*value).unwrap()
        });
        Ok(())
    }

    pub fn write(&mut self,idx:usize,value:Complex32) -> Result<(),CflError> {
        self.write_slice(idx, &[value])?;
        Ok(())
    }

    pub fn write_slice(&mut self,dest_idx:usize,src:&[Complex32]) -> Result<(),CflError> {
        if dest_idx + src.len() > self.n_elements {
            Err(CflError::Mmap)?
        }
        let start = dest_idx*size_of::<Complex32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *mut u8,
                self.mmap[start..(start+size_of::<Complex32>()*src.len())].as_mut_ptr(),
                src.len() * size_of::<Complex32>(),
            );
        }
        self.mmap.flush().map_err(|_|CflError::MmapFlush)?;
        Ok(())
    }

    pub fn write_op_from<F>(&mut self,dst_indices:&[usize],src:&[Complex32],op:F) -> Result<(),CflError>
    where F: Fn(Complex32,Complex32) -> Complex32 {
        src.iter().zip(dst_indices.iter()).for_each(|(value,dst_idx)|{
            self.write_op(*dst_idx,*value,&op).unwrap()
        });
        self.mmap.flush().map_err(|_|CflError::MmapFlush)?;
        Ok(())
    }

    pub fn write_op<F>(&mut self,idx:usize,value:Complex32,op:F) -> Result<(),CflError>
    where F: Fn(Complex32,Complex32) -> Complex32 {
        let mut s = [Complex32::ZERO];
        self.write_op_slice(idx, &[value], &mut s, op)?;
        Ok(())
    }

    /// write src to file starting at dest_idx while performing some operation on previous values
    pub fn write_op_slice<F>(&mut self,dest_idx:usize,src:&[Complex32],scratch_buff:&mut [Complex32], op:F) -> Result<(),CflError>
    where F: Fn(Complex32,Complex32) -> Complex32 {
        if dest_idx + src.len() > self.n_elements {
            Err(CflError::Mmap)?
        }

        assert_eq!(src.len(),scratch_buff.len());

        let start = dest_idx*size_of::<Complex32>();

        //let mut buff = vec![Complex32::ZERO;src.len()];

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mmap[start..(start+size_of::<Complex32>()*src.len())].as_ptr(),
                scratch_buff.as_mut_ptr() as *mut u8,
                src.len() * size_of::<Complex32>(),
            );
        }

        scratch_buff.iter_mut().zip(src.iter()).for_each(|(dst,src)|{
            *dst = op(*dst,*src)
        });

        unsafe {
            std::ptr::copy_nonoverlapping(
                scratch_buff.as_ptr() as *mut u8,
                self.mmap[start..(start+size_of::<Complex32>()*src.len())].as_mut_ptr(),
                src.len() * size_of::<Complex32>(),
            );
        }

        Ok(())
    }

    pub fn flush(&self) -> Result<(),CflError> {
        self.mmap.flush().map_err(|_|CflError::MmapFlush)
    }

}



impl fmt::Display for CflError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl Error for CflError {}

pub fn get_dims<T: AsRef<Path>>(file_path: T) -> Result<Vec<usize>, CflError> {
    let mut f =
        File::open(file_path.as_ref().with_extension("hdr")).map_err(|err| CflError::IO(err))?;
    let mut s = String::new();
    f.read_to_string(&mut s).map_err(|err| CflError::IO(err))?;
    let re = Regex::new(r"Dimensions").unwrap();
    let lines: Vec<_> = s.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        if line.starts_with("#") && re.is_match(line) {
            if let Some(next) = lines.get(idx + 1) {
                let dims: Vec<_> = next
                    .split_whitespace()
                    .map(|s| s.parse::<usize>().unwrap())
                    .collect();
                return Ok(dims);
            }
        }
    }
    Err(CflError::DimensionsNotFound)
}

/// write cfl header with dimensions
fn write_header<T: AsRef<Path>>(file_path: T, dims: &[usize]) -> Result<(), CflError> {
    if dims.len() > 16 {
        Err(CflError::TooManyDims(dims.len()))?
    }

    let mut f = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(file_path.as_ref().with_extension("hdr"))
        .map_err(|err| CflError::IO(err))?;

    let dims_str = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let hdr_contents = format!("# Dimensions\n{}\n", dims_str);
    f.write_all(hdr_contents.as_bytes())
        .map_err(|err| CflError::IO(err))?;

    Ok(())
}


/// blindly tries to fill the buffer with file contents. This is efficient if you already
/// have a properly sized buffer for the file, otherwize, use read_cfl_to_vec()
fn memmap_cfl_to_buff(f: &File, buff: &mut [Complex32]) -> io::Result<()> {
    let mmap = unsafe { MmapOptions::new().map(f)? };
    unsafe {
        std::ptr::copy_nonoverlapping(
            mmap.as_ptr(),
            buff.as_mut_ptr() as *mut u8,
            buff.len() * std::mem::size_of::<Complex32>(),
        );
    }
    Ok(())
}

/// write contents of buff to a file using memory map. The file must be set to read/write
fn unmap_buff_to_cfl(f: &File, buff: &[Complex32]) -> io::Result<()> {
    let n_bytes = buff.len() * std::mem::size_of::<Complex32>();
    f.set_len(n_bytes as u64)?;
    let mut mmap = unsafe { MmapMut::map_mut(f)? };
    let data_bytes = unsafe { std::slice::from_raw_parts(buff.as_ptr() as *const u8, n_bytes) };
    mmap.copy_from_slice(data_bytes);
    Ok(())
}

pub fn to_array<T: AsRef<Path>>(
    file_path: T,
    ignore_singletons: bool,
) -> Result<ArrayD<Complex32>, CflError> {
    let dims: Vec<_> = if ignore_singletons {
        get_dims(&file_path)?
            .into_iter()
            .filter(|x| *x != 1)
            .collect()
    } else {
        get_dims(&file_path)?
    };

    //let mut buff = ArrayD::zeros(dims.as_slice().set_f(true));
    let mut buff = ArrayD::zeros(dims.as_slice().f());

    let file =
        File::open(file_path.as_ref().with_extension("cfl")).map_err(|err| CflError::IO(err))?;
    memmap_cfl_to_buff(&file, buff.as_slice_memory_order_mut().unwrap())
        .map_err(|err| CflError::IO(err))?;
    Ok(buff)
}

pub fn from_array<T: AsRef<Path>>(file_path: T, x: &ArrayD<Complex32>) -> Result<(), CflError> {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path.as_ref().with_extension("cfl"))
        .unwrap();

    let buff = x.as_slice_memory_order().ok_or(CflError::InvalidMemOrder)?;
    unmap_buff_to_cfl(&f, buff).map_err(|err| CflError::IO(err))?;

    write_header(file_path, x.shape())?;

    Ok(())
}

/// check if both the data file and header exist and are read-able
pub fn exists<T: AsRef<Path>>(filename: T) -> Result<(), CflError> {
    let data_prop = filename.as_ref().with_extension("cfl");
    let data_header = filename.as_ref().with_extension("hdr");
    if !data_prop.exists() {
        Err(CflError::CflNotFound(data_prop))?
    }
    if !data_header.exists() {
        Err(CflError::CflNotFound(data_header))?
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::time::Instant;
    use num_traits::Zero;

    #[test]
    fn read_to_array() {
        //cargo test --release --package cfl --lib -- tests::read_to_array --exact --nocapture

        let dims = [788, 480, 480];
        let n_elems: usize = dims.iter().product();

        let data = rand_array(&[n_elems])
            .as_slice_memory_order()
            .unwrap()
            .to_vec();

        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open("cfl_test.cfl")
            .unwrap();

        println!("writing array ...");
        unmap_buff_to_cfl(&f, &data).unwrap();
        write_header("cfl_test", &dims).unwrap();

        println!("reading to array ...");
        let now = Instant::now();
        let arr = to_array("cfl_test", true).unwrap();
        println!("read took: {} ms", now.elapsed().as_millis());

        let s = arr.as_slice_memory_order().unwrap();

        println!("comparing ...");
        s.iter()
            .zip(data.iter())
            .for_each(|pair| assert_eq!(pair.0, pair.1));

        std::fs::remove_file("cfl_test.cfl").unwrap();
        std::fs::remove_file("cfl_test.hdr").unwrap();

        println!("done.");
    }

    #[test]
    fn read_and_write() {
        //cargo test --release --package cfl --lib -- tests::read_and_write --exact --nocapture

        let n_elems = 788 * 480 * 480;

        println!("filling array ...");
        let mut rng = rand::thread_rng();
        let mut data = Vec::<Complex32>::with_capacity(n_elems);
        for _ in 0..n_elems {
            data.push(Complex32::new(
                rng.gen_range((-1.)..1.),
                rng.gen_range((-1.)..1.),
            ))
        }

        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open("cfl_test")
            .unwrap();

        println!("writing array ...");
        unmap_buff_to_cfl(&f, &data).unwrap();

        let mut new_buff = vec![Complex32::ZERO; n_elems];
        println!("reading array ...");
        memmap_cfl_to_buff(&f, &mut new_buff).unwrap();

        println!("comparing ...");
        new_buff
            .iter()
            .zip(data.iter())
            .for_each(|pair| assert!((pair.0 - pair.1).is_zero()));

        println!("cleaning up ...");
        std::fs::remove_file("cfl_test").unwrap();

        println!("done.");
    }

    #[test]
    fn read_and_write_header() {
        let dims = [30, 45, 235, 23];

        write_header("test_cfl", &dims).unwrap();

        let out_dims = get_dims("test_cfl").unwrap();

        for (d1, d2) in dims.into_iter().zip(out_dims.into_iter()) {
            assert_eq!(d1, d2)
        }
    }

    fn rand_array(dims: &[usize]) -> ArrayD<Complex32> {
        let n_elems = dims.iter().product();
        let mut rng = rand::thread_rng();
        let mut data = Vec::<Complex32>::with_capacity(n_elems);
        for _ in 0..n_elems {
            data.push(Complex32::new(
                rng.gen_range((-1.)..1.),
                rng.gen_range((-1.)..1.),
            ))
        }
        ArrayD::from_shape_vec(dims, data).unwrap()
    }
}
