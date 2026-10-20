//! The `medh5` command line.
//!
//! One native application over the format engine: the same grammar, output and
//! exit codes as the 1.x Python command line (0 success, 1 a handled error, 2
//! a usage error), whether it runs as the standalone `medh5` binary or inside
//! the Python package's console script.  The format converters are Python
//! integrations and run in the Python package (see [`Host`]).

use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Stdio};

use clap::error::ErrorKind;
use serde_json::{json, Map, Value};

use medh5::json::repr_str;

mod app;
mod bench;
mod clinical;
mod common;
mod conformance;
mod convert;
mod curation;
mod dataset;
mod inspect;
mod labels;
mod perf;
mod resample;
mod scrub;
mod seg;
mod workers;

use common::Ctx;
pub use common::{sample_file_refusal, top_level_message, Host, EXIT_ERROR, EXIT_OK, EXIT_USAGE};

/// The host of the standalone binary: hands the converters and the dataloader
/// benchmark to a Python interpreter that has the `medh5` package, when there
/// is one (`MEDH5_PYTHON`, else `python3`, or `python` on Windows).
pub struct NativeHost;

impl NativeHost {
    fn python() -> String {
        std::env::var("MEDH5_PYTHON").unwrap_or_else(|_| if cfg!(windows) { "python".into() } else { "python3".into() })
    }
}

/// What every program the binary hands to Python runs first.  `python -c`
/// and `python -m` put the current directory at the head of `sys.path`, so
/// a `json.py` --- or a `medh5/` --- where `medh5 convert` runs would be
/// imported, and run, in place of the real one; a downloaded study folder is
/// where converters run.  `-P` does this only from Python 3.11, and `-I`
/// would also drop the user's site-packages and `PYTHONPATH`.
const SAFE_PATH: &str = "import sys; sys.path[:1] == [''] and sys.path.pop(0)";

/// `python -c CODE`, the current directory off `sys.path` before anything
/// is imported; arguments added after it are `sys.argv[1:]`.
fn python_running(python: &str, code: &str) -> Command {
    let mut command = Command::new(python);
    command.arg("-c").arg(format!("{SAFE_PATH}; {code}"));
    command
}

impl Host for NativeHost {
    fn convert(
        &self,
        argv: &[String],
        _command: &str,
        _args: &Value,
        _out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Option<i32> {
        hand_over(&Self::python(), argv, err)
    }

    fn throughput(&self, path: &str, patch: i64, workers: i64, annotation: Option<&str>) -> Result<Value, String> {
        measure_throughput(&Self::python(), path, patch, workers, annotation)
    }

    /// This binary again.
    fn worker_command(&self) -> Option<Vec<OsString>> {
        std::env::current_exe().ok().map(|exe| vec![exe.into_os_string()])
    }
}

/// `command.output()`, started through [`medh5::h5::spawn`] so that the child
/// inherits no file this process has open.
fn output(command: &mut Command) -> std::io::Result<std::process::Output> {
    medh5::h5::spawn(command.stdout(Stdio::piped()).stderr(Stdio::piped()))?.wait_with_output()
}

/// The last line `bytes` hold that is not blank: of a traceback, the
/// exception (`ModuleNotFoundError: No module named 'medh5'`).
fn last_line(bytes: &[u8]) -> Option<String> {
    String::from_utf8_lossy(bytes).lines().map(str::trim).rfind(|l| !l.is_empty()).map(str::to_string)
}

/// Whether `python` can be handed a command: it runs, and imports `modules`
/// from a `medh5` of this one's major version --- the 1.x package's
/// `medh5.cli` has another grammar, and its converters write another
/// format.  `Err` says why not, in Python's words where Python gave some.
fn probe(python: &str, modules: &str) -> Result<(), String> {
    let code = format!("import medh5, {modules}; print(medh5.__version__)");
    let ran = output(python_running(python, &code).stdin(Stdio::null()))
        .map_err(|e| format!("{python} could not be run: {e}"))?;
    if !ran.status.success() {
        let why = last_line(&ran.stderr).unwrap_or_else(|| ran.status.to_string());
        return Err(format!("{python} could not import {modules}: {why}"));
    }
    let found = last_line(&ran.stdout).unwrap_or_default();
    let major = medh5::VERSION.split('.').next().unwrap_or_default();
    if found.split('.').next() != Some(major) {
        return Err(format!(
            "{python} has medh5 {found}, and medh5 {} hands its commands only to medh5 {major}.x",
            medh5::VERSION
        ));
    }
    Ok(())
}

/// Run `medh5 argv` in `python`'s package; `None`, with the reason on `err`,
/// when it cannot.
fn hand_over(python: &str, argv: &[String], err: &mut dyn Write) -> Option<i32> {
    if let Err(why) = probe(python, "medh5.io, medh5.cli") {
        let _ = writeln!(err, "medh5: {why}");
        let _ = err.flush();
        return None;
    }
    let status =
        medh5::h5::spawn(python_running(python, "from medh5.cli import main; raise SystemExit(main())").args(argv))
            .and_then(|mut child| child.wait())
            .ok()?;
    Some(status.code().unwrap_or(EXIT_ERROR))
}

/// The dataloader benchmark, run by `python`'s `medh5` package as the console
/// script runs it: `Measurement.to_json()`, or why it could not run there.
fn measure_throughput(
    python: &str,
    path: &str,
    patch: i64,
    workers: i64,
    annotation: Option<&str>,
) -> Result<Value, String> {
    const CODE: &str = "import json; from medh5.bench import throughput; print(json.dumps(throughput([sys.argv[1]], \
                        patch=int(sys.argv[2]), workers=int(sys.argv[3]), annotation=sys.argv[4] or None).to_json()))";
    let needs = format!(
        "the dataloader benchmark runs in the Python package, with PyTorch (pip install 'medh5[torch]'); {python} \
         could not run it (MEDH5_PYTHON names another interpreter)"
    );
    // A 1.x package would measure its own dataloader under this one's name.
    probe(python, "medh5.bench").map_err(|why| format!("{needs}: {why}"))?;
    let ran = output(
        python_running(python, CODE)
            .args([path, &patch.to_string(), &workers.to_string(), annotation.unwrap_or("")])
            .stdin(Stdio::null()),
    )
    .map_err(|e| format!("{needs}: {e}"))?;
    if !ran.status.success() {
        // The last line of a traceback is the exception: `No module named 'torch'`.
        return Err(format!("{needs}: {}", last_line(&ran.stderr).unwrap_or_else(|| ran.status.to_string())));
    }
    let printed = last_line(&ran.stdout).unwrap_or_default();
    serde_json::from_str(&printed).map_err(|e| format!("{needs}: it printed no measurement ({e})"))
}

/// Run the command line on `args` (without the program name) against the
/// process's standard streams.
pub fn run(args: Vec<String>) -> i32 {
    let mut out = std::io::stdout().lock();
    let mut err = std::io::stderr().lock();
    run_with(&args, &mut out, &mut err, &NativeHost)
}

/// [`run`] on arguments as the operating system passed them, which need not
/// be UTF-8: one that is not is a usage error, said on stderr.
pub fn run_os(args: impl IntoIterator<Item = OsString>) -> i32 {
    match utf8_args(args) {
        Ok(args) => run(args),
        Err(message) => {
            let _ = writeln!(std::io::stderr(), "medh5: {message}");
            EXIT_USAGE
        }
    }
}

/// The arguments as text, or what to say about the first that is not UTF-8
/// --- a Latin-1 file name a shell glob expanded, say.  Every command takes
/// its paths as text, so such an argument is refused before anything runs
/// rather than ending the program in a panic.
pub fn utf8_args(args: impl IntoIterator<Item = OsString>) -> Result<Vec<String>, String> {
    args.into_iter()
        .map(|arg| {
            arg.into_string().map_err(|arg| format!("argument {} is not valid UTF-8", repr_str(&arg.to_string_lossy())))
        })
        .collect()
}

/// Run the command line on `args` (without the program name); the exit code.
pub fn run_with(args: &[String], out: &mut dyn Write, err: &mut dyn Write, host: &dyn Host) -> i32 {
    medh5::h5::init();
    // A worker of a `--workers` run: not a command anyone types, so not in
    // the grammar.
    if args.first().map(String::as_str) == Some("__worker") {
        let _ = err.flush();
        return workers::serve(&args[1..], &mut std::io::stdin().lock(), out);
    }
    let argv = std::iter::once("medh5".to_string()).chain(args.iter().cloned());
    let mut command = app::command();
    let mut ctx = Ctx::new(out, err, host);
    let matches = match command.try_get_matches_from_mut(argv) {
        Ok(m) => m,
        Err(e) => {
            let text = e.render().to_string();
            let code = match e.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    ctx.print(text.trim_end());
                    EXIT_OK
                }
                _ => {
                    ctx.eprint(text.trim_end());
                    EXIT_USAGE
                }
            };
            return ctx.finish(code);
        }
    };
    // No command is a usage error, so its help goes where every usage error
    // goes, as a command group's does: stdout is for help asked for.
    let Some((name, sub)) = matches.subcommand() else {
        let help = command.render_help();
        ctx.eprint(help.to_string().trim_end());
        return ctx.finish(EXIT_USAGE);
    };
    let result = match name {
        "info" | "tree" | "validate" | "verify" | "fix" | "timeline" | "track" => {
            inspect::dispatch(name, sub, &mut ctx)
        }
        "seg" | "index" => seg::dispatch(name, sub, &mut ctx),
        "labels" => labels::dispatch(sub, &mut ctx),
        "pack" | "unpack" | "ls" | "prov" | "agree" | "scrub" | "splits" => curation::dispatch(name, sub, &mut ctx),
        "dataset" => dataset::dispatch(sub, &mut ctx),
        "convert" => convert::convert(args, sub, &mut ctx),
        "migrate" => convert::migrate(sub, &mut ctx),
        "recompress" | "bench" => perf::dispatch(name, sub, &mut ctx),
        "resample" => resample::run(sub, &mut ctx),
        "conformance" => conformance::dispatch(sub, &mut ctx),
        "clinical" | "task" | "cache" => clinical::dispatch(args, name, sub, &mut ctx),
        _ => Ok(EXIT_USAGE),
    };
    let code = match result {
        Ok(code) => code,
        Err(e) => {
            let message = top_level_message(&e);
            ctx.fail(message)
        }
    };
    ctx.finish(code)
}

/// The grammar as data: `{"options": [...], "positionals": [...],
/// "commands": {name: {...}}}`, for tools that check documentation against it.
pub fn command_tree() -> Value {
    fn walk(cmd: &clap::Command) -> Value {
        let mut options = Vec::new();
        let mut positionals = Vec::new();
        for arg in cmd.get_arguments() {
            if arg.is_positional() {
                positionals.push(json!(arg.get_id().as_str()));
                continue;
            }
            for short in arg.get_short_and_visible_aliases().unwrap_or_default() {
                options.push(json!(format!("-{short}")));
            }
            for long in arg.get_long_and_visible_aliases().unwrap_or_default() {
                options.push(json!(format!("--{long}")));
            }
        }
        let mut commands = Map::new();
        for sub in cmd.get_subcommands() {
            commands.insert(sub.get_name().to_string(), walk(sub));
        }
        json!({"options": options, "positionals": positionals, "commands": commands})
    }
    // Built, the grammar has the `--help` and `--version` clap adds.
    let mut command = app::command();
    command.build();
    walk(&command)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn an_argument_that_is_not_utf8_is_a_usage_error_not_a_panic() {
        use std::os::unix::ffi::OsStringExt;

        let latin1 = || OsString::from_vec(b"caf\xe9.medh5".to_vec());
        let refused = utf8_args([OsString::from("info"), latin1()]).unwrap_err();
        assert_eq!(refused, "argument 'caf\u{fffd}.medh5' is not valid UTF-8");
        assert_eq!(run_os([OsString::from("info"), latin1()]), EXIT_USAGE);
        assert_eq!(utf8_args([OsString::from("info"), OsString::from("café.medh5")]).unwrap(), ["info", "café.medh5"]);
    }

    #[test]
    fn python_handed_a_program_imports_nothing_from_the_current_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("json.py"), "raise SystemExit(99)\n").unwrap();
        let python = NativeHost::python();
        let planted = |command: &mut Command| command.current_dir(dir.path()).env_remove("PYTHONSAFEPATH").status();
        // Without the guard the planted module is what runs: the attack is real here.
        let Ok(unguarded) = planted(Command::new(&python).args(["-c", "import json"])) else {
            return; // no Python to hand anything to
        };
        assert_eq!(unguarded.code(), Some(99));
        let guarded = planted(&mut python_running(&python, "import json; json.dumps([])")).unwrap();
        assert_eq!(guarded.code(), Some(0));
    }

    /// A host that records what it is handed, and runs it when `runs`.
    struct Recording {
        runs: bool,
        handed: std::cell::RefCell<Vec<(String, Value)>>,
    }

    impl Host for Recording {
        fn convert(
            &self,
            argv: &[String],
            command: &str,
            args: &Value,
            _: &mut dyn Write,
            _: &mut dyn Write,
        ) -> Option<i32> {
            assert_eq!(argv.first().map(String::as_str), Some("clinical"));
            self.handed.borrow_mut().push((command.to_string(), args.clone()));
            self.runs.then_some(EXIT_OK)
        }

        fn throughput(&self, _: &str, _: i64, _: i64, _: Option<&str>) -> Result<Value, String> {
            unreachable!()
        }
    }

    #[test]
    fn w22_02_the_nifti_converters_run_without_python() {
        use medh5::array::NdArray;
        use medh5::sample::{create, GridOptions, ImageOptions};
        use std::path::Path;

        let dir = tempfile::tempdir().unwrap();
        let at = |name: &str| dir.path().join(name).to_string_lossy().into_owned();
        let ct = NdArray::from_vec(&[4, 5, 6], (0..120).map(|v| v as i16).collect()).unwrap();
        let mut w = create(Path::new(&at("source.medh5")), Some("s"), None, "portable", &[]).unwrap();
        w.add_grid("g", &[4, 5, 6], &[2.0, 1.0, 1.0], GridOptions::default()).unwrap();
        w.add_image("CT", &ct, "g", "CT", ImageOptions::default()).unwrap();
        w.commit(true).unwrap();
        // No Python: a command handed over would fail, and be recorded.
        let host = Recording { runs: false, handed: Default::default() };
        let run = |args: &[&str]| -> (i32, String, String) {
            let argv: Vec<String> = args.iter().map(|a| a.to_string()).collect();
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let code = run_with(&argv, &mut out, &mut err, &host);
            (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
        };
        let (code, out, err) = run(&["convert", "to-nifti", &at("source.medh5"), "CT", &at("ct.nii.gz")]);
        assert_eq!(code, EXIT_OK, "{err}");
        assert_eq!(out.trim_end(), at("ct.nii.gz"), "an export prints the file it wrote");
        let image = format!("CT={}", at("ct.nii.gz"));
        let back = at("back.medh5");
        let (code, _, err) = run(&["convert", "from-nifti", &back, "--image", &image, "--report", &at("r.json")]);
        assert_eq!(code, EXIT_OK, "{err}");
        let report: Value = serde_json::from_str(&std::fs::read_to_string(at("r.json")).unwrap()).unwrap();
        assert_eq!(report["converter"], "from-nifti");
        let organ = format!("organ={}", at("ct.nii.gz"));
        let (code, _, err) = run(&["convert", "import-seg-nifti", &back, "--mask", &organ]);
        assert_eq!(code, EXIT_OK, "{err}");
        assert!(medh5::sample::open_sample(Path::new(&back)).unwrap().annotation("seg").is_ok());
        // The checks the Python package made before reading anything.
        let (code, _, err) = run(&["convert", "from-nifti", &at("x.medh5"), "--image", "CT"]);
        assert_eq!((code, err.trim_end()), (EXIT_ERROR, "medh5: --image expects NAME=VALUE, got 'CT'"));
        let (code, _, err) =
            run(&["convert", "from-nifti", &at("x.medh5"), "--image", &image, "--report", &at("ct.nii.gz")]);
        let clash = format!(
            "medh5: --report {} is image 'CT', which this command reads; name another file for the report",
            at("ct.nii.gz")
        );
        assert_eq!((code, err.trim_end()), (EXIT_ERROR, clash.as_str()));
        let (code, _, err) =
            run(&["convert", "to-nifti", &at("source.medh5"), "CT", &at("x.nii.gz"), "--report", &at("")]);
        assert_eq!(code, EXIT_ERROR);
        assert!(err.ends_with("is a directory; name the file to write\n"), "{err}");
        assert!(host.handed.take().is_empty(), "nothing was handed to Python");
    }

    #[test]
    fn s10_the_importers_are_handed_to_python() {
        let argv = |rest: &[&str]| -> Vec<String> {
            ["clinical", "augment", "s.medh5"].iter().chain(rest).map(|a| a.to_string()).collect()
        };
        let host = Recording { runs: true, handed: Default::default() };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let meds =
            argv(&["--from-meds", "meds", "--meds-subject", "1001", "--kind", "LOINC=observation", "--with-text"]);
        assert_eq!(run_with(&meds, &mut out, &mut err, &host), EXIT_OK);
        let fhir = argv(&["--from-fhir", "b.json", "--fhir-subject", "Patient/p1", "--timezone", "UTC"]);
        assert_eq!(run_with(&fhir, &mut out, &mut err, &host), EXIT_OK);
        let handed = host.handed.take();
        assert_eq!(handed.len(), 2);
        let (command, args) = &handed[0];
        assert_eq!(command, "clinical-augment");
        assert_eq!(args["from_meds"], "meds");
        assert_eq!(args["meds_subject"], 1001);
        assert_eq!(args["kind"], serde_json::json!(["LOINC=observation"]));
        assert_eq!(args["with_text"], true);
        assert_eq!(args["records"], Value::Null);
        assert_eq!(handed[1].1["fhir_subject"], "Patient/p1");
        assert_eq!(handed[1].1["timezone"], "UTC");
        // No Python to hand them to: what to install, and exit 1.
        let host = Recording { runs: false, handed: Default::default() };
        let mut err = Vec::new();
        assert_eq!(run_with(&fhir, &mut out, &mut err, &host), EXIT_ERROR);
        assert!(String::from_utf8(err).unwrap().contains("MEDH5_PYTHON"));
        assert_eq!(host.handed.take().len(), 1);
        // A logical-record bundle never leaves the engine, and takes no
        // importer option.
        let mut err = Vec::new();
        let records = argv(&["records.json", "--timezone", "UTC"]);
        assert_eq!(run_with(&records, &mut out, &mut err, &host), EXIT_USAGE);
        assert!(host.handed.take().is_empty());
    }

    #[test]
    fn s10_a_cohort_import_is_handed_to_python() {
        let argv = |rest: &[&str]| -> Vec<String> {
            ["clinical", "augment-cohort"].iter().chain(rest).map(|a| a.to_string()).collect()
        };
        let host = Recording { runs: true, handed: Default::default() };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let meds = argv(&["--from-meds", "meds", "--mapping", "map.csv", "--out-dir", "out", "--kind", "LOINC=lab"]);
        assert_eq!(run_with(&meds, &mut out, &mut err, &host), EXIT_OK);
        let fhir = argv(&["--from-fhir-bulk", "export", "--mapping", "map.csv", "--timezone", "UTC"]);
        assert_eq!(run_with(&fhir, &mut out, &mut err, &host), EXIT_OK);
        let handed = host.handed.take();
        assert_eq!(handed.len(), 2);
        let (command, args) = &handed[0];
        assert_eq!(command, "clinical-augment-cohort");
        assert_eq!(args["from_meds"], "meds");
        assert_eq!(args["mapping"], "map.csv");
        assert_eq!(args["out_dir"], "out");
        assert_eq!(args["kind"], serde_json::json!(["LOINC=lab"]));
        assert_eq!(handed[1].1["from_fhir_bulk"], "export");
        assert_eq!(handed[1].1["timezone"], "UTC");
        // One source, and a mapping, or a usage error before Python runs.
        for wrong in [
            argv(&["--mapping", "map.csv"]),
            argv(&["--from-meds", "meds"]),
            argv(&["--from-meds", "meds", "--from-fhir-bulk", "export", "--mapping", "map.csv"]),
        ] {
            assert_eq!(run_with(&wrong, &mut out, &mut err, &host), EXIT_USAGE, "{wrong:?}");
        }
        assert!(host.handed.take().is_empty());
        // No Python to hand it to: what to install, and exit 1.
        let host = Recording { runs: false, handed: Default::default() };
        let mut err = Vec::new();
        assert_eq!(run_with(&fhir, &mut out, &mut err, &host), EXIT_ERROR);
        assert!(String::from_utf8(err).unwrap().contains("MEDH5_PYTHON"));
    }

    /// A stdout every write to which fails with `error()`.
    struct Failing(fn() -> std::io::Error);

    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err((self.0)())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn run_into(stdout: &mut dyn Write, args: &[&str]) -> (i32, String) {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        let mut err = Vec::new();
        let code = run_with(&args, stdout, &mut err, &NativeHost);
        (code, String::from_utf8(err).unwrap())
    }

    #[test]
    fn output_lost_to_a_full_disk_fails_the_command() {
        let mut full = Failing(|| std::io::Error::other("No space left on device"));
        for args in [&["conformance", "list"][..], &["conformance", "list", "--json"], &["--help"]] {
            let (code, err) = run_into(&mut full, args);
            assert_eq!(code, EXIT_ERROR, "{args:?}");
            assert_eq!(err, "medh5: cannot write output: No space left on device\n", "{args:?}");
        }
    }

    #[test]
    fn a_reader_that_left_early_keeps_the_commands_own_exit_code() {
        let mut gone = Failing(|| std::io::ErrorKind::BrokenPipe.into());
        assert_eq!(run_into(&mut gone, &["conformance", "list"]), (EXIT_OK, String::new()));
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.medh5");
        let (code, err) = run_into(&mut gone, &["validate", missing.to_str().unwrap()]);
        assert_eq!((code, err.as_str()), (EXIT_ERROR, ""));
    }

    #[test]
    fn a_throughput_run_python_cannot_make_says_what_it_needs() {
        let why = measure_throughput("/nonexistent/python3", "case.medh5", 64, 0, None).unwrap_err();
        assert!(why.contains("pip install 'medh5[torch]'") && why.contains("MEDH5_PYTHON"), "{why}");
    }

    /// The probe discarded what Python printed, so "not installed" was all a
    /// broken install ever said; and it accepted any `medh5`, so a 1.x
    /// package was handed 2.x commands --- another grammar, and converters
    /// that write another format.
    #[cfg(unix)]
    #[test]
    fn a_python_is_handed_a_command_only_by_a_package_of_this_major() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        // A stand-in interpreter: `-c CODE ARGS...`, the probe's code being
        // the one that prints the version.
        let fake = |name: &str, on_probe: &str| {
            let path = dir.path().join(name);
            let body = format!(
                "#!/bin/sh\ncase \"$2\" in\n  *__version__*) {on_probe} ;;\n  *) shift 2; echo \"$@\" > \"{}\"; exit 3 \
                 ;;\nesac\n",
                dir.path().join("handed").display()
            );
            std::fs::write(&path, body).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            // A file just written can be busy (ETXTBSY) for as long as a
            // process another test forked meanwhile still holds it open.
            for _ in 0..200 {
                match Command::new(&path).args(["-c", "__version__"]).output() {
                    Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                        std::thread::sleep(std::time::Duration::from_millis(5))
                    }
                    _ => break,
                }
            }
            path.to_string_lossy().into_owned()
        };
        let major = medh5::VERSION.split('.').next().unwrap();
        let current = fake("current", &format!("echo {}", medh5::VERSION));
        let old = fake("old", "echo 1.4.2");
        let broken = fake(
            "broken",
            "printf 'Traceback (most recent call last):\\nModuleNotFoundError: No module named medh5\\n\\n' >&2; exit 1",
        );
        let modules = "medh5.io, medh5.cli";

        assert_eq!(probe(&current, modules), Ok(()));
        let refused =
            format!("{old} has medh5 1.4.2, and medh5 {} hands its commands only to medh5 {major}.x", medh5::VERSION);
        assert_eq!(probe(&old, modules), Err(refused.clone()));
        assert_eq!(
            probe(&broken, modules),
            Err(format!("{broken} could not import {modules}: ModuleNotFoundError: No module named medh5"))
        );
        let missing = probe("/nonexistent/python3", modules).unwrap_err();
        assert!(missing.starts_with("/nonexistent/python3 could not be run: "), "{missing}");

        // Handed over: the command line as given, and its exit code.
        let argv = ["convert".to_string(), "from-dicom".to_string(), "out.medh5".to_string()];
        let mut err = Vec::new();
        assert_eq!(hand_over(&current, &argv, &mut err), Some(3));
        assert!(err.is_empty(), "{}", String::from_utf8_lossy(&err));
        let handed = std::fs::read_to_string(dir.path().join("handed")).unwrap();
        assert_eq!(handed.trim_end(), "convert from-dicom out.medh5");
        // Refused: nothing run, and why on stderr.
        std::fs::remove_file(dir.path().join("handed")).unwrap();
        assert_eq!(hand_over(&old, &argv, &mut err), None);
        assert_eq!(String::from_utf8(err).unwrap(), format!("medh5: {refused}\n"));
        assert!(!dir.path().join("handed").exists());
        let why = measure_throughput(&old, "case.medh5", 64, 0, None).unwrap_err();
        assert!(why.ends_with(&refused) && why.contains("pip install 'medh5[torch]'"), "{why}");
    }
}
