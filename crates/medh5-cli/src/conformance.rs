//! `conformance` --- build the corpus, run it here, publish it, score others.

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::conformance::{
    build_corpus, case_by_name, cases, check_checksums, load_manifest, publish, run_case, run_corpus, score_with,
    summarize, write_expected, CaseResult, ScoreOptions,
};
use medh5::json::repr_str;
use medh5::pyval::py_path;
use medh5::Error;

use crate::common::*;
use crate::workers;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let names = m.subcommand().and_then(|(_, sub)| sub.try_get_many::<String>("names").ok().flatten());
    let names: Option<Vec<String>> = names.map(|n| n.cloned().collect());
    let chosen = match chosen(names.as_deref()) {
        Ok(n) => n,
        Err(why) => return Ok(ctx.usage(why)),
    };
    let names = names.as_deref();
    match m.subcommand() {
        Some(("list", sub)) => list(sub, ctx),
        Some(("build", sub)) if flag(sub, "robustness") => {
            let dir = outdir(sub)?;
            let written = medh5::conformance::robustness::write(dir)?;
            ctx.print(format!("wrote {} robustness fixtures to {}", written.len(), py_path(dir)));
            Ok(EXIT_OK)
        }
        Some(("build", sub)) => {
            let manifest = build_corpus(outdir(sub)?, names)?;
            ctx.print(format!("wrote {chosen} cases and {}", py_path(&manifest)));
            Ok(EXIT_OK)
        }
        Some(("run", sub)) => {
            let results = if workers::wanted(sub) > 1 {
                run_in_units(sub, names, ctx)?
            } else {
                run_corpus(outdir(sub)?, names)?
            };
            Ok(report(&results, None, None, flag(sub, "json"), ctx))
        }
        Some(("publish", sub)) => {
            let root = py_path(&publish(outdir(sub)?, names)?);
            ctx.print(format!("wrote the suite to {root}: {chosen} cases, see {root}/README.md"));
            Ok(EXIT_OK)
        }
        Some(("score", sub)) => score_cmd(sub, ctx),
        _ => Ok(ctx.usage("usage: medh5 conformance {list|build|run|publish|score}")),
    }
}

/// `run_corpus`, a case to a unit on `--workers` processes: each builds and
/// checks its case, and the manifest is written once every case is built.
fn run_in_units(m: &ArgMatches, names: Option<&[String]>, ctx: &mut Ctx) -> medh5::Result<Vec<CaseResult>> {
    let dir = outdir(m)?;
    let units: Vec<workers::Unit> = cases()
        .iter()
        .filter(|c| names.is_none_or(|n| n.contains(&c.name)))
        .map(|c| {
            let path = dir.join(format!("{}{}", c.name, c.suffix));
            workers::Unit {
                task: json!({"name": c.name, "path": py_path(&path)}),
                files: vec![path.to_string_lossy().into_owned()],
                label: c.name.clone(),
            }
        })
        .collect();
    let mut results = Vec::with_capacity(units.len());
    workers::run_units("conformance run", m, ctx, &units, &mut |_, outcome| {
        results.push(result_from_json(&outcome.data)?);
        Ok(())
    })?;
    write_expected(dir, names)?;
    Ok(results)
}

/// `conformance run --workers`: a unit is a case.  A case whose worker
/// stopped could not be run, as a case whose file the validator cannot
/// read could not.
pub(crate) const RUN: workers::Kind = workers::Kind {
    runner: |m| Ok(Box::new(CaseRunner { dir: outdir(m)?.to_path_buf() })),
    died: |unit, status| {
        let case = case_by_name(unit.task["name"].as_str().unwrap_or_default())?;
        let mut result = CaseResult::new(case.clone(), unit.task["path"].as_str().unwrap_or_default().to_string());
        result.error = Some(format!("the worker running it stopped ({status})"));
        let mut outcome = workers::FileOutcome::new();
        outcome.data = result_to_json(&result);
        Ok(outcome)
    },
    writes: true,
};

struct CaseRunner {
    dir: std::path::PathBuf,
}

impl workers::Runner for CaseRunner {
    fn run(&mut self, task: &Value) -> medh5::Result<workers::FileOutcome> {
        let result = run_case(&self.dir, case_by_name(task["name"].as_str().unwrap_or_default())?)?;
        let mut outcome = workers::FileOutcome::new();
        outcome.data = result_to_json(&result);
        Ok(outcome)
    }
}

fn result_to_json(r: &CaseResult) -> Value {
    json!({
        "name": r.case.name,
        "path": r.path,
        "got_errors": r.got_errors,
        "got_warnings": r.got_warnings,
        "missing": r.missing,
        "unexpected": r.unexpected,
        "error": r.error,
        "details": r.details,
    })
}

fn result_from_json(v: &Value) -> medh5::Result<CaseResult> {
    let strings = |key: &str| -> Vec<String> {
        v[key].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect()
    };
    let case = case_by_name(v["name"].as_str().unwrap_or_default())?;
    let mut result = CaseResult::new(case.clone(), v["path"].as_str().unwrap_or_default().to_string());
    result.got_errors = strings("got_errors");
    result.got_warnings = strings("got_warnings");
    result.missing = strings("missing");
    result.unexpected = strings("unexpected");
    result.error = v["error"].as_str().map(str::to_string);
    result.details = strings("details");
    Ok(result)
}

/// `OUTDIR`, made if it is not there; a file in the way is named.
fn outdir(m: &ArgMatches) -> medh5::Result<&Path> {
    let dir = req_str(m, "outdir");
    std::fs::create_dir_all(dir).map_err(|e| medh5::error::os_error(&e, dir))?;
    Ok(Path::new(dir))
}

/// How many cases `--case` selects --- every case when it names none --- or
/// why it selects nothing: a name no case has would build, run or publish an
/// empty corpus and report it a success.
fn chosen(names: Option<&[String]>) -> Result<usize, String> {
    if let Some(unknown) = names.into_iter().flatten().find(|n| case_by_name(n).is_err()) {
        return Err(format!("no conformance case {} (`medh5 conformance list` names them)", repr_str(unknown)));
    }
    Ok(cases().iter().filter(|c| names.is_none_or(|n| n.contains(&c.name))).count())
}

fn list(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if flag(m, "json") {
        let listing: Vec<Value> = cases().iter().map(|c| c.to_json()).collect();
        ctx.emit(&Value::Array(listing), true);
        return Ok(EXIT_OK);
    }
    let rows: Vec<Vec<String>> = cases()
        .iter()
        .map(|c| {
            let mut codes: Vec<&str> = c.errors.iter().chain(&c.warnings).map(String::as_str).collect();
            codes.sort_unstable();
            vec![
                c.name.clone(),
                c.clause.clone(),
                c.level.clone(),
                if c.valid() { "valid" } else { "invalid" }.to_string(),
                if codes.is_empty() { "-".to_string() } else { codes.join(",") },
            ]
        })
        .collect();
    ctx.print(table(&rows, &["case", "clause", "level", "kind", "expected codes"]));
    Ok(EXIT_OK)
}

fn score_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let suite = Path::new(req_str(m, "suite"));
    let results_path = Path::new(req_str(m, "results"));
    let companion = flag(m, "companion");
    if companion && (get_str(m, "format_version").is_some() || get_i64(m, "codes_revision").is_some()) {
        return Ok(ctx
            .usage("--companion scores the task-and-cache fixtures, which no format version or code revision bounds"));
    }
    let mut options = ScoreOptions::default();
    if let Some(version) = get_str(m, "format_version") {
        options = options.format_version(version);
    }
    if let Some(revision) = get_i64(m, "codes_revision") {
        options = options.codes_revision(revision as u32);
    }
    let scored = (|| -> Result<(Vec<String>, Vec<CaseResult>, usize), Error> {
        let text =
            std::fs::read_to_string(results_path).map_err(|e| medh5::error::os_error(&e, py_path(results_path)))?;
        let submitted = medh5::json::loads(&text).map_err(|e| Error::Value(format!("not valid JSON: {e}")))?;
        let stale = check_checksums(suite)?;
        let entries: Vec<Value> = match submitted {
            Value::Array(items) => items,
            // Iterating a mapping yields its keys; none of them is a result.
            Value::Object(map) => map.keys().map(|k| Value::String(k.clone())).collect(),
            other => vec![other],
        };
        if companion {
            let results = medh5::conformance::companion::score(suite, &entries)?;
            let listed = results.len();
            return Ok((stale, results, listed));
        }
        let results = score_with(suite, &entries, &options)?;
        let listed = load_manifest(suite)?.get("cases").and_then(Value::as_array).map_or(0, Vec::len);
        Ok((stale, results, listed))
    })();
    let (stale, results, listed) = match scored {
        Ok(found) => found,
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_) | Error::Value(_)) => return Ok(ctx.fail(e.python_str())),
        Err(e) => return Err(e),
    };
    let as_json = flag(m, "json");
    if !stale.is_empty() {
        // The scores mean nothing if the files scored are not the files
        // published, so say so before reporting any of them --- on stderr
        // when stdout is the document, which names them under `changed`.
        let mut warning =
            vec![format!("WARNING: {} published file(s) differ from {}", stale.len(), req_str(m, "suite"))];
        warning.extend(stale.iter().map(|name| format!("  changed: {name}")));
        for line in warning {
            if as_json {
                ctx.eprint(line);
            } else {
                ctx.print(line);
            }
        }
    }
    let skipped = listed.saturating_sub(results.len());
    Ok(report(&results, Some(&stale), Some(skipped), as_json, ctx))
}

/// Print the outcome; the exit code.  `changed`, for a score, names the
/// published files whose bytes are no longer the published ones: a score
/// over them is not a score, so it fails whatever the cases say.
/// `skipped` counts the cases a bounded score left out.
fn report(
    results: &[CaseResult],
    changed: Option<&[String]>,
    skipped: Option<usize>,
    as_json: bool,
    ctx: &mut Ctx,
) -> i32 {
    let failures: Vec<&CaseResult> = results.iter().filter(|r| !r.ok()).collect();
    let drifted = changed.is_some_and(|c| !c.is_empty());
    if as_json {
        let mut summary = summarize(results);
        if let Some(changed) = changed {
            summary["changed"] = json!(changed);
            summary["ok"] = json!(failures.is_empty() && !drifted);
        }
        if let Some(skipped) = skipped {
            summary["skipped"] = json!(skipped);
        }
        ctx.emit(&summary, true);
    } else {
        for result in &failures {
            ctx.print(format!("FAIL {}", result.case.name));
            if let Some(error) = &result.error {
                ctx.print(format!("     {error}"));
            }
            if !result.missing.is_empty() {
                ctx.print(format!("     not reported: {}", result.missing.join(", ")));
            }
            if !result.unexpected.is_empty() {
                ctx.print(format!("     unexpected:   {}", result.unexpected.join(", ")));
            }
        }
        ctx.print(format!("{}/{} cases pass", results.len() - failures.len(), results.len()));
        if let Some(skipped @ 1..) = skipped {
            ctx.print(format!("{skipped} case(s) above the bounds given were not scored"));
        }
    }
    if failures.is_empty() && !drifted {
        EXIT_OK
    } else {
        EXIT_ERROR
    }
}
