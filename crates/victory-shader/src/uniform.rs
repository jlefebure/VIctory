//! Which values may differ between the lanes of a console wave.
//!
//! A console wave64 program runs on a 32-lane host as two host waves. Most
//! of its code does not care: each lane computes its own values. A
//! `v_readfirstlane_b32` does: on the console it gives every lane the first
//! active lane of the 64, on the host the first active lane of its own 32.
//! The two agree only when the value read is the same in every active lane,
//! which the console compiler usually knows (a value computed from scalars
//! but held in a VGPR). [`first_lane_reads_vary`] finds the reads where that
//! is not shown: the program then needs the console's 64-lane waves.

use std::collections::HashSet;

use crate::ir::{EXEC, Program, Reg, Value};
use crate::structure::{Atom, LaneExit, Structured};

/// Values that may differ between lanes: VGPR inputs other than
/// `uniform_inputs`, the lane index (`v_mbcnt`), and whatever is computed
/// from them, including VGPR values that join after a per-lane branch.
///
/// EXEC is not followed (it selects lanes, it is no lane's value), nor is
/// the old value of a VGPR an unguarded instruction overwrites (no running
/// lane keeps it).
pub fn lane_varying(p: &Program, st: &Structured, uniform_inputs: &[Reg]) -> HashSet<Value> {
    let mut varying: HashSet<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| matches!(r, Reg::V(_)) && !uniform_inputs.contains(r))
        .map(|&(_, v)| v)
        .collect();
    let per_lane_exit = |b: usize| match st.exits[b] {
        LaneExit::Branch { when, .. } => !st
            .bools
            .support(when)
            .iter()
            .all(|a| matches!(a, Atom::Uniform(_))),
        _ => false,
    };
    loop {
        let before = varying.len();
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                let from_varying = phi.args.iter().any(|a| varying.contains(a));
                let differ = phi.args.iter().any(|a| *a != phi.args[0]);
                let per_lane_join = matches!(phi.reg, Reg::V(_))
                    && differ
                    && block.preds.iter().any(|&q| per_lane_exit(q));
                if from_varying || per_lane_join {
                    varying.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let unguarded = st.guards[b][i].is_none();
                let reads_varying = inst.reads.iter().any(|&(r, v)| {
                    let exec = matches!(r, Reg::S(n) if n == EXEC || n == EXEC + 1);
                    let overwritten =
                        unguarded && matches!(r, Reg::V(_)) && inst.write(r).is_some();
                    !exec && !overwritten && varying.contains(&v)
                });
                if reads_varying || inst.name().starts_with("v_mbcnt") {
                    varying.extend(inst.writes.iter().map(|&(_, v)| v));
                }
            }
        }
        if varying.len() == before {
            return varying;
        }
    }
}

/// Whether some `v_readfirstlane_b32` may read a value that differs between
/// the active lanes (see the module documentation).
pub fn first_lane_reads_vary(p: &Program, st: &Structured, uniform_inputs: &[Reg]) -> bool {
    let varying = lane_varying(p, st, uniform_inputs);
    p.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
        inst.name() == "v_readfirstlane_b32"
            && inst
                .reads
                .iter()
                .any(|&(r, v)| matches!(r, Reg::V(_)) && varying.contains(&v))
    })
}

#[cfg(test)]
mod tests {
    use super::first_lane_reads_vary;
    use crate::ir::Reg;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn reads_vary(words: &[u32]) -> bool {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        first_lane_reads_vary(&p, &st, &[Reg::V(5)])
    }

    #[test]
    fn a_value_from_scalars_reads_alike_in_every_lane() {
        assert!(!reads_vary(&[
            0x7E04_0505, // v_readfirstlane_b32 s2, v5 (a uniform input)
            0x7E02_0202, // v_mov_b32 v1, s2
            0x0602_0281, // v_add_f32 v1, 1, v1
            0x7E06_0501, // v_readfirstlane_b32 s3, v1
            S_ENDPGM,
        ]));
    }

    #[test]
    fn a_value_from_the_lane_index_does_not() {
        assert!(reads_vary(&[
            0xD765_0000,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
            0xD766_0000,
            0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
            0x7E06_0500, // v_readfirstlane_b32 s3, v0
            S_ENDPGM,
        ]));
    }

    #[test]
    fn a_vgpr_input_other_than_the_uniform_ones_may_vary() {
        assert!(reads_vary(&[
            0x7E06_0501, // v_readfirstlane_b32 s3, v1
            S_ENDPGM,
        ]));
    }
}
