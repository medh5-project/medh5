//! `resample` --- images onto another grid, as a recorded derivation (1.0
//! §10.2, §4.4, §11.1).

use std::path::Path;

use clap::ArgMatches;

use medh5::array::DType;
use medh5::transforms::resample::{resample, ResamplePlan};

use crate::common::*;

pub fn run(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let images = get_strs(m, "image").unwrap_or_default();
    let spacing = m.get_one::<Vec<f64>>("spacing").cloned();
    let onto = get_str(m, "onto").map(str::to_string);
    let grid_id = get_str(m, "grid_id").map(str::to_string);
    if grid_id.is_some() && spacing.is_none() {
        return Ok(ctx.usage("--grid-id names the new grid --spacing makes, so it goes with --spacing, not --onto"));
    }
    let mut plan = ResamplePlan::from_options(images, onto, spacing, grid_id)?
        .with_interpolation(req_str(m, "interpolation"))
        .with_fill(get_f64(m, "fill").unwrap_or(0.0));
    if let Some(suffix) = get_str(m, "suffix") {
        plan = plan.with_suffix(suffix);
    }
    if let Some(dtype) = get_str(m, "dtype") {
        match DType::parse(dtype) {
            Ok(d) => plan = plan.with_dtype(d),
            Err(_) => return Ok(ctx.usage(format!("--dtype {dtype} is not a dtype the format stores"))),
        }
    }
    if let Some(out) = get_str(m, "out") {
        plan = plan.with_out(out);
    }
    if flag(m, "no_annotations") {
        plan = plan.with_annotations(false);
    }
    let result = resample(Path::new(path), &plan).map_err(|e| sample_file_error(path, e))?;
    if flag(m, "json") {
        ctx.emit(&result.to_json(), true);
    } else {
        for line in result.lines() {
            ctx.print(line);
        }
    }
    Ok(EXIT_OK)
}
