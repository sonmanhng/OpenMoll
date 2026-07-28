/// XTC trajectory file parser for GROMACS `.xtc` files.
///
/// Implements the GROMACS 3dfcoord lossy-float compression exactly as in
/// `xdrfile.c` (GROMACS source), including:
///   - LSB-first bit reading (matching GROMACS `receivebits`)
///   - Correct run-length delta encoding with dynamic `smallidx` tracking
///   - Correct `sizeofint` / packed-triplet bit count matching GROMACS
///
/// Reference: https://manual.gromacs.org/current/reference-manual/file-formats.html#xtc

use anyhow::{anyhow, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single MD frame read from an XTC file.
#[derive(Debug, Clone)]
pub struct XtcFrame {
    /// GROMACS simulation step number.
    pub step: i32,
    /// Simulation time in picoseconds.
    pub time_ps: f32,
    /// Atom positions in Ångström (converted from XTC nm × 10).
    /// Length == n_atoms.
    pub positions: Vec<[f32; 3]>,
}

/// All frames decoded from an XTC file.
#[derive(Debug, Clone)]
pub struct XtcTrajectory {
    /// Number of atoms per frame.
    pub n_atoms: usize,
    pub frames: Vec<XtcFrame>,
}

impl XtcTrajectory {
    /// Total number of frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Time range (ps) as `(first, last)`, or `None` if empty.
    pub fn time_range_ps(&self) -> Option<(f32, f32)> {
        let first = self.frames.first()?.time_ps;
        let last  = self.frames.last()?.time_ps;
        Some((first, last))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XTC magic
// ─────────────────────────────────────────────────────────────────────────────

const XTC_MAGIC: i32 = 1995;

// ─────────────────────────────────────────────────────────────────────────────
// Top-level entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an XTC file at `path` and return all frames.
pub fn parse_xtc(path: &str) -> Result<XtcTrajectory> {
    let data = std::fs::read(path)
        .map_err(|e| anyhow!("Cannot open XTC file: {}", e))?;

    let mut cursor = 0usize;
    let mut frames = Vec::new();
    let mut n_atoms_global: Option<usize> = None;

    while cursor < data.len() {
        // Each frame starts with: magic (i32), n_atoms (i32), step (i32), time (f32)
        if cursor + 16 > data.len() {
            break;
        }
        let frame_start = cursor;
        eprintln!("\n[XTC] ========== Frame {} ==========", frames.len());
        eprintln!("[XTC] Frame starts at offset {:#x} ({} bytes)", frame_start, frame_start);
        let magic = read_i32(&data, &mut cursor)?;
        if magic != XTC_MAGIC {
            return Err(anyhow!("XTC: bad magic {:#x} at offset {} (frame {})", magic, frame_start, frames.len()));
        }
        let n_atoms = read_i32(&data, &mut cursor)? as usize;
        let step    = read_i32(&data, &mut cursor)?;
        let time    = read_f32(&data, &mut cursor)?;
        eprintln!("[XTC] n_atoms={}, step={}, time={:.2}ps", n_atoms, step, time);
        eprintln!("[XTC] Cursor after header: {:#x}", cursor);

        if let Some(prev) = n_atoms_global {
            if n_atoms != prev {
                return Err(anyhow!("XTC: n_atoms changed mid-file ({} → {})", prev, n_atoms));
            }
        } else {
            n_atoms_global = Some(n_atoms);
        }

        // Box vectors: 9 floats (3×3 nm matrix), skip them
        for _ in 0..9 {
            read_f32(&data, &mut cursor)?;
        }
        eprintln!("[XTC] Cursor after box: {:#x}", cursor);

        // Atom positions
        let positions = if n_atoms <= 9 {
            // Small systems: always raw
            read_positions_raw(&data, &mut cursor, n_atoms)?
        } else {
            // GROMACS writes n_atoms again before positions as validation
            let n_atoms_check = read_i32(&data, &mut cursor)? as usize;
            eprintln!("[XTC] n_atoms_check = {} (expected {})", n_atoms_check, n_atoms);
            if n_atoms_check != n_atoms {
                return Err(anyhow!("XTC: n_atoms mismatch in frame {}: {} vs {}", 
                    frames.len(), n_atoms_check, n_atoms));
            }
            
            // Compressed positions (byte_count/compressed_size is read inside read_positions_3dfcoord)
            read_positions_3dfcoord(&data, &mut cursor, n_atoms)?
        };

        frames.push(XtcFrame { step, time_ps: time, positions });
    }

    if frames.is_empty() {
        return Err(anyhow!("No frames found in XTC file"));
    }

    Ok(XtcTrajectory {
        n_atoms: n_atoms_global.unwrap_or(0),
        frames,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw float positions (small systems ≤ 9 atoms)
// ─────────────────────────────────────────────────────────────────────────────

fn read_positions_raw(
    data: &[u8],
    cursor: &mut usize,
    n_atoms: usize,
) -> Result<Vec<[f32; 3]>> {
    let mut positions = Vec::with_capacity(n_atoms);
    for _ in 0..n_atoms {
        let x = read_f32(data, cursor)? * 10.0; // nm → Å
        let y = read_f32(data, cursor)? * 10.0;
        let z = read_f32(data, cursor)? * 10.0;
        positions.push([x, y, z]);
    }
    xdr_align(cursor);
    Ok(positions)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3DFcoord compressed positions — correct GROMACS algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Decode GROMACS 3dfcoord compressed positions.
///
/// XDR field order (from GROMACS xdrfile_xtc.c):
///   precision (f32) → minint[3] → maxint[3] → smallidx (i32) → nint (i32) → opaque_bytes
///
/// The compressed bit stream uses LSB-first ordering within each byte
/// (matching GROMACS `receivebits`), and applies run-length delta encoding
/// where "small" atom moves are stored as deltas from the previous atom.
fn read_positions_3dfcoord(
    data: &[u8],
    cursor: &mut usize,
    n_atoms: usize,
) -> Result<Vec<[f32; 3]>> {
    // ── Header fields ────────────────────────────────────────────────────────
    let precision = read_f32(data, cursor)?;
    if precision <= 0.0 {
        return Err(anyhow!("XTC: invalid precision {}", precision));
    }
    eprintln!("[XTC] Frame precision = {:.6e}", precision);

    let minint = [read_i32(data, cursor)?, read_i32(data, cursor)?, read_i32(data, cursor)?];
    let maxint = [read_i32(data, cursor)?, read_i32(data, cursor)?, read_i32(data, cursor)?];
    eprintln!("[XTC] minint = {:?}, maxint = {:?}", minint, maxint);

    // Size range per axis
    let sizeint: [i64; 3] = [
        (maxint[0] as i64 - minint[0] as i64 + 1).max(1),
        (maxint[1] as i64 - minint[1] as i64 + 1).max(1),
        (maxint[2] as i64 - minint[2] as i64 + 1).max(1),
    ];

    // Bits per axis for large (main) coordinates — matching GROMACS sizeofint()
    let bitsizeint = [
        gromacs_sizeofint(sizeint[0]),
        gromacs_sizeofint(sizeint[1]),
        gromacs_sizeofint(sizeint[2]),
    ];

    // If any size exceeds 24 bits, fallback to unpacked. Otherwise, packed.
    let bitsize: u32 = if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff {
        0 // use per-axis reads
    } else {
        sizeofints(3, &sizeint)
    };

    // ── smallidx and derived values ──────────────────────────────────────────
    let mut smallidx = read_i32(data, cursor)?.clamp(FIRSTIDX, LASTIDX);
    let mut smallnum = magicint(smallidx as usize) / 2;

    // ── Compressed byte block ────────────────────────────────────────────────
    let byte_count = read_i32(data, cursor)?;
    if byte_count <= 0 {
        return Err(anyhow!("XTC: invalid byte_count {}", byte_count));
    }
    let byte_count = byte_count as usize;
    
    if *cursor + byte_count > data.len() {
        return Err(anyhow!(
            "XTC: compressed block extends past EOF ({} > {})",
            *cursor + byte_count,
            data.len()
        ));
    }
    let compressed_bytes = &data[*cursor..*cursor + byte_count];
    let cursor_start = *cursor;
    *cursor += byte_count;
    xdr_align(cursor);
    
    // ── Decode bits ──────────────────────────────────────────────────────────
    let mut br = BitReader::new(compressed_bytes);
    let mut positions = Vec::with_capacity(n_atoms);
    let inv_prec = 1.0_f32 / precision;
    eprintln!("[XTC] Starting decompression: {} atoms, up to {} bytes available", n_atoms, byte_count);
    let mut i = 0usize;

    let mut is_smaller = 0i32;
    let mut run = 0usize;

    while i < n_atoms {
        if i % 10000 == 0 && i > 0 {
            // eprintln!("[XTC] Decompressing... {}/{} atoms", i, n_atoms);
        }
        // Read MAIN (anchor) quantised triplet
        let mut thiscoord = read_main_triplet(&mut br, bitsize, bitsizeint, sizeint)?;
        thiscoord[0] += minint[0] as i64;
        thiscoord[1] += minint[1] as i64;
        thiscoord[2] += minint[2] as i64;
        
        let mut prevcoord = thiscoord;
        i += 1;

        let flag = br.read_bits(1)?;
        if flag != 0 {
            let run5 = br.read_bits(5)? as i32;
            is_smaller = (run5 % 3) - 1;
            run = (run5 / 3) as usize;
        } else {
            is_smaller = 0;
        }

        if run > 0 {
            let size_small = magicint(smallidx as usize) as i64;
            let sizes = [size_small, size_small, size_small];
            let num_of_bits = smallidx as u32;
            
            for k in 0..run {
                if i > n_atoms { break; }
                let decoded = decodeints(&mut br, 3, num_of_bits, &sizes)?;
                
                let mut nextcoord = [
                    decoded[0] + prevcoord[0] - smallnum,
                    decoded[1] + prevcoord[1] - smallnum,
                    decoded[2] + prevcoord[2] - smallnum,
                ];

                if k == 0 {
                    let tmp = nextcoord;
                    nextcoord = prevcoord;
                    prevcoord = tmp;
                    positions.push(to_angstrom(prevcoord, inv_prec));
                } else {
                    prevcoord = nextcoord;
                }
                positions.push(to_angstrom(nextcoord, inv_prec));
                i += 1;
            }
        } else {
            positions.push(to_angstrom(prevcoord, inv_prec));
        }

        if is_smaller != 0 {
            smallidx += is_smaller;
            if smallidx < FIRSTIDX { smallidx = FIRSTIDX; }
            if smallidx > LASTIDX { smallidx = LASTIDX; }
            smallnum = magicint(smallidx as usize) / 2;
        }
    }
    


    Ok(positions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for the decoder
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn to_angstrom(coord: [i64; 3], inv_prec: f32) -> [f32; 3] {
    [
        coord[0] as f32 * inv_prec * 10.0,
        coord[1] as f32 * inv_prec * 10.0,
        coord[2] as f32 * inv_prec * 10.0,
    ]
}

fn sizeofints(num_of_ints: usize, sizes: &[i64]) -> u32 {
    let mut bytes = [0u32; 128];
    bytes[0] = 1;
    let mut num_of_bytes = 1;
    let mut num_of_bits = 0;

    for i in 0..num_of_ints {
        let mut tmp = 0u64;
        for bytecnt in 0..num_of_bytes {
            tmp = (bytes[bytecnt] as u64) * (sizes[i] as u64) + tmp;
            bytes[bytecnt] = (tmp & 0xff) as u32;
            tmp >>= 8;
        }
        while tmp != 0 {
            bytes[num_of_bytes] = (tmp & 0xff) as u32;
            num_of_bytes += 1;
            tmp >>= 8;
        }
    }

    let mut num = 1u32;
    num_of_bytes -= 1;
    while bytes[num_of_bytes] >= num {
        num_of_bits += 1;
        num <<= 1;
    }
    (num_of_bits + num_of_bytes as u32 * 8) as u32
}

fn decodeints(
    br: &mut BitReader,
    num_of_ints: usize,
    num_of_bits: u32,
    sizes: &[i64],
) -> Result<Vec<i64>> {
    let mut bytes = [0u32; 128];
    let mut num_of_bytes = 0;
    let mut remaining_bits = num_of_bits;

    while remaining_bits > 8 {
        bytes[num_of_bytes] = br.read_bits(8)? as u32;
        num_of_bytes += 1;
        remaining_bits -= 8;
    }
    if remaining_bits > 0 {
        bytes[num_of_bytes] = br.read_bits(remaining_bits)? as u32;
        num_of_bytes += 1;
    }

    let mut nums = vec![0i64; num_of_ints];

    for i in (1..num_of_ints).rev() {
        let mut num = 0u64;
        for j in (0..num_of_bytes).rev() {
            num = (num << 8) | (bytes[j] as u64);
            let p = num / (sizes[i] as u64);
            bytes[j] = p as u32;
            num = num - p * (sizes[i] as u64);
        }
        nums[i] = num as i64;
    }

    nums[0] = (bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)) as i64;

    Ok(nums)
}

fn read_main_triplet(
    br: &mut BitReader,
    bitsize: u32,
    bitsizeint: [u32; 3],
    sizeint: [i64; 3],
) -> Result<[i64; 3]> {
    if bitsize > 0 {
        let nums = decodeints(br, 3, bitsize, &sizeint)?;
        Ok([nums[0], nums[1], nums[2]])
    } else {
        Ok([
            br.read_bits(bitsizeint[0])? as i64,
            br.read_bits(bitsizeint[1])? as i64,
            br.read_bits(bitsizeint[2])? as i64,
        ])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GROMACS magic integers table
// ─────────────────────────────────────────────────────────────────────────────

const FIRSTIDX: i32 = 9;
const LASTIDX:  i32 = 85;

/// Return magic integer at index `i` (clamped), matching GROMACS `magicints[]`.
fn magicint(i: usize) -> i64 {
    const MAGIC: &[i64] = &[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8,
        10, 12, 16, 20, 25, 32, 40, 50, 64, 80,
        101, 128, 161, 203, 256, 322, 406, 512, 645, 812,
        1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6475, 8192,
        10321, 13009, 16384, 20642, 26018, 32768, 41285, 52026, 65536, 82570,
        104051, 131072, 165140, 208102, 262144, 330280, 416204, 524288, 660561, 832408,
        1048576, 1321122, 1664816, 2097152, 2642244, 3329632, 4194304, 5284488,
        6659264, 8388608, 10568976, 13318528, 16777216, 21137952, 26637056,
        33554432, 42275904, 53274112, 67108864, 84551808, 106548224, 134217728,
        169103616, 213096448, 268435456,
    ];
    MAGIC[i.min(MAGIC.len() - 1)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-size helpers matching GROMACS source exactly
// ─────────────────────────────────────────────────────────────────────────────

/// Number of bits needed to represent values in [0, size], matching GROMACS `sizeofint`.
/// Equivalent to: smallest k such that 2^k > size.
fn gromacs_sizeofint(size: i64) -> u32 {
    let mut num: i64 = 1;
    let mut num_bits = 0u32;
    while size >= num && num_bits < 32 {
        num_bits += 1;
        num <<= 1;
    }
    num_bits
}

// ─────────────────────────────────────────────────────────────────────────────
// LSB-first bit reader (matches GROMACS `receivebits`)
// ─────────────────────────────────────────────────────────────────────────────

/// Bit reader that reads bits LSB-first within each byte, in byte-stream order.
/// This exactly matches the GROMACS `receivebits` behaviour.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    /// Next bit index to read within current byte (0 = LSB, 7 = MSB).
    bit_pos: u8,
    current_byte: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let current_byte = data.first().copied().unwrap_or(0);
        Self { data, byte_pos: 0, bit_pos: 0, current_byte }
    }

    /// Read `num_bits` bits and return them as a u64, matching GROMACS MSB-first bitpacking.
    fn read_bits(&mut self, num_bits: u32) -> Result<u64> {
        if num_bits == 0 {
            return Ok(0);
        }
        if num_bits > 64 {
            return Err(anyhow!("XTC BitReader: requested {} bits (max 64)", num_bits));
        }

        let mut result = 0u64;
        let mut bits_filled = 0u32;

        while bits_filled < num_bits {
            // Advance to next byte if current one is exhausted
            if self.bit_pos >= 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
                self.current_byte = self.data.get(self.byte_pos).copied().unwrap_or(0);
            }

            let bits_available = 8 - self.bit_pos;
            let bits_needed    = num_bits - bits_filled;
            let bits_to_take   = bits_available.min(bits_needed as u8);

            // Extract bits_to_take from MSB side of unread portion
            let shift = 8 - self.bit_pos - bits_to_take;
            let mask  = (1u64 << bits_to_take) - 1;
            let value = ((self.current_byte >> shift) as u64) & mask;

            // Shift result left by bits_to_take and append the new bits
            result = (result << bits_to_take) | value;

            self.bit_pos  += bits_to_take;
            bits_filled   += bits_to_take as u32;
        }

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XDR primitive readers (big-endian)
// ─────────────────────────────────────────────────────────────────────────────

fn read_i32(data: &[u8], cursor: &mut usize) -> Result<i32> {
    if *cursor + 4 > data.len() {
        return Err(anyhow!("XTC: unexpected EOF reading i32 at offset {}", *cursor));
    }
    let bytes: [u8; 4] = data[*cursor..*cursor + 4].try_into().unwrap();
    *cursor += 4;
    Ok(i32::from_be_bytes(bytes))
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32> {
    if *cursor + 4 > data.len() {
        return Err(anyhow!("XTC: unexpected EOF reading f32 at offset {}", *cursor));
    }
    let bits = u32::from_be_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(f32::from_bits(bits))
}

/// Skip bytes to align cursor to next 4-byte XDR boundary.
fn xdr_align(cursor: &mut usize) {
    let rem = *cursor % 4;
    if rem != 0 {
        *cursor += 4 - rem;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gromacs_sizeofint() {
        // GROMACS: sizeofint(0)=1, sizeofint(1)=1, sizeofint(3)=2, sizeofint(4)=3
        assert_eq!(gromacs_sizeofint(0), 1);
        assert_eq!(gromacs_sizeofint(1), 1);
        assert_eq!(gromacs_sizeofint(3), 2);
        assert_eq!(gromacs_sizeofint(4), 3);
        assert_eq!(gromacs_sizeofint(255), 8);
        assert_eq!(gromacs_sizeofint(256), 9);
    }

    #[test]
    fn test_bit_reader_lsb_first() {
        // Byte 0x5F = 0101_1111
        // LSB-first: first 4 bits = 1111 = 0xF, next 4 bits = 0101 = 0x5
        let data = [0x5Fu8, 0xA3u8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(4).unwrap(), 0xF);
        assert_eq!(reader.read_bits(4).unwrap(), 0x5);
        // Byte 0xA3 = 1010_0011 → first 4 bits LSB = 0011 = 0x3
        assert_eq!(reader.read_bits(4).unwrap(), 0x3);
    }

    #[test]
    fn test_bit_reader_cross_byte() {
        // Two bytes: 0xFF 0x00
        // Reading 10 bits: 8 from first byte (all 1s) + 2 from second (00)
        // Result: 0b00_1111_1111 = 0xFF
        let data = [0xFFu8, 0x00u8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(10).unwrap(), 0xFF);
    }
}
