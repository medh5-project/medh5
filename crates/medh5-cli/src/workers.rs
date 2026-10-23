//! `--workers N`: the work of a batch command across processes.
//!
//! One HDF5 lock covers each process, filters included, so threads in one
//! process take turns on every read and decode; parallelism belongs to
//! processes.  The parent starts up to `N` workers --- this program again,
//! as [`Host::worker_command`] names it, running the hidden `__worker`
//! command --- and hands each a unit at a time: a JSON line `{"index",
//! "task"}` on its standard input, answered by one line holding the unit's
//! [`FileOutcome`].  The parent takes the outcomes in input order, so the
//! text, the `--json` document and the exit code are a sequential run's,
//! byte for byte, whatever the number of workers.
//!
//! A unit is a file for the commands that take their files one by one
//! (`validate`, `verify`, `fix`, `index build`, `recompress`), and for the
//! others what the command computes without looking at another: a subject's
//! samples (`migrate`), a subject's rows (`task preflight`), a case
//! (`conformance run`).  What the units share is computed once, by the
//! parent, before any worker starts, and travels in each task.
//!
//! An error that stops a sequential run crosses as its kind and message, is
//! raised again at the same unit, and stops this run there.  Units that
//! read or write one file --- the paths a writing command is given that name
//! one file, the inputs and output of a sample --- go to one worker, in
//! order.  A worker that dies fails only the unit it held, as its command
//! words such a failure; what it had not begun goes to the others, and when
//! none is left, the parent runs it.  The parent opens no file before the
//! workers start: a child must not inherit an open HDF5 file.
//!
//! [`Host::worker_command`]: crate::Host::worker_command

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;

use clap::parser::ValueSource;
use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::Error;

use crate::common::*;

/// Which stream a line of a file's output goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// What one file of a batch command produced: the lines it printed, in
/// order, its entry in the `--json` document, and whether it passed.
/// `data` is whatever else the command's summary reads (the rows of
/// `recompress`'s table).
#[derive(Clone, Debug, PartialEq)]
pub struct FileOutcome {
    pub lines: Vec<(Stream, String)>,
    pub entry: Value,
    pub data: Value,
    pub ok: bool,
}

impl FileOutcome {
    pub fn new() -> FileOutcome {
        FileOutcome { lines: Vec::new(), entry: Value::Null, data: Value::Null, ok: true }
    }

    /// `print(text)`, kept for the parent to print in its turn.
    pub fn print(&mut self, text: impl Into<String>) {
        self.lines.push((Stream::Out, text.into()));
    }

    /// A file that could not be processed, as [`Ctx::file_failed`] reports
    /// one: `medh5: why` on stderr, and the entry the `--json` document
    /// gives it.
    pub fn failed(path: &str, error: &Error) -> FileOutcome {
        let message = in_cli_terms(&top_level_message(error));
        FileOutcome {
            lines: vec![(Stream::Err, format!("medh5: {message}"))],
            entry: json!({"path": path, "ok": false, "error": message}),
            data: Value::Null,
            ok: false,
        }
    }

    fn to_wire(&self) -> Value {
        let lines: Vec<Value> = self.lines.iter().map(|(s, t)| json!([*s == Stream::Err, t])).collect();
        json!({"lines": lines, "entry": self.entry, "data": self.data, "ok": self.ok})
    }

    fn from_wire(value: &Value) -> Option<FileOutcome> {
        let lines = value
            .get("lines")?
            .as_array()?
            .iter()
            .map(|line| {
                let stream = if line.get(0)?.as_bool()? { Stream::Err } else { Stream::Out };
                Some((stream, line.get(1)?.as_str()?.to_string()))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(FileOutcome {
            lines,
            entry: value.get("entry")?.clone(),
            data: value.get("data")?.clone(),
            ok: value.get("ok")?.as_bool()?,
        })
    }
}

impl Default for FileOutcome {
    fn default() -> Self {
        FileOutcome::new()
    }
}

/// One file of a command: what it prints and reports, or an error that
/// stops the run.
pub type One = fn(&str, &ArgMatches) -> medh5::Result<FileOutcome>;

/// One unit of a command's work, as the parent hands it out.
#[derive(Clone)]
pub struct Unit {
    /// What the worker is handed.
    pub task: Value,
    /// The files it reads or writes: for a command that writes, units that
    /// share one go to one worker, in input order.
    pub files: Vec<String>,
    /// How a message names it: a path, a case, a subject.
    pub label: String,
}

/// What runs a command's units in one process: made once per process from
/// the command's arguments, then handed each task in turn.
pub trait Runner {
    fn run(&mut self, task: &Value) -> medh5::Result<FileOutcome>;
}

/// A command that takes `--workers`.
#[derive(Clone, Copy)]
pub struct Kind {
    /// How a process makes its [`Runner`].
    pub runner: fn(&ArgMatches) -> medh5::Result<Box<dyn Runner>>,
    /// What stands for a unit whose worker stopped without answering, given
    /// the worker's exit status: as the command words a unit that failed,
    /// or the error that stops the run.
    pub died: fn(&Unit, &str) -> medh5::Result<FileOutcome>,
    /// Whether its units write files.
    pub writes: bool,
}

/// A command whose units are its files.
struct Files {
    one: One,
    m: ArgMatches,
}

impl Runner for Files {
    fn run(&mut self, task: &Value) -> medh5::Result<FileOutcome> {
        (self.one)(task.as_str().unwrap_or_default(), &self.m)
    }
}

fn files(one: One, m: &ArgMatches) -> medh5::Result<Box<dyn Runner>> {
    Ok(Box::new(Files { one, m: m.clone() }))
}

/// A file a worker stopped on fails, and the run goes on.
fn file_died(unit: &Unit, status: &str) -> medh5::Result<FileOutcome> {
    let error = Error::Runtime(format!("{}: the worker running it stopped ({status})", unit.label));
    Ok(FileOutcome::failed(&unit.label, &error))
}

/// The commands that take `--workers`, by their words.
fn command(words: &str) -> Option<Kind> {
    let file = |runner, writes| Kind { runner, died: file_died, writes };
    Some(match words {
        "validate" => file(|m| files(crate::inspect::validate_one, m), false),
        "verify" => file(|m| files(crate::inspect::verify_one, m), false),
        "fix" => file(|m| files(crate::inspect::fix_one, m), true),
        "index build" => file(|m| files(crate::seg::index_build_one, m), true),
        "recompress" => file(|m| files(crate::perf::recompress_one, m), true),
        "migrate" => crate::convert::MIGRATE,
        "task preflight" => crate::clinical::PREFLIGHT,
        "conformance run" => crate::conformance::RUN,
        _ => return None,
    })
}

/// The number of processes `--workers` asks for: at least one.
pub fn wanted(m: &ArgMatches) -> usize {
    get_i64(m, "workers").unwrap_or(1).max(1) as usize
}

/// Run `words`'s file function over `paths`, on `--workers` processes when
/// it asks for more than one and the host can start them; print each file's
/// lines as its turn comes, and return the outcomes in input order.  An
/// error stops the run at its file, as it stops a sequential one.
pub fn run_files(words: &str, m: &ArgMatches, ctx: &mut Ctx, paths: &[String]) -> medh5::Result<Vec<FileOutcome>> {
    let units: Vec<Unit> =
        paths.iter().map(|p| Unit { task: json!(p), files: vec![p.clone()], label: p.clone() }).collect();
    let mut outcomes = Vec::with_capacity(paths.len());
    run_units(words, m, ctx, &units, &mut |ctx, outcome| {
        replay(ctx, &outcome);
        outcomes.push(outcome);
        Ok(())
    })?;
    Ok(outcomes)
}

/// Run `words`'s `units` on `--workers` processes when it asks for more than
/// one and the host can start them, else here; hand each outcome to `take`
/// in input order.  An error --- a unit's, or `take`'s --- stops the run at
/// its unit, as it stops a sequential one.
pub fn run_units(
    words: &str,
    m: &ArgMatches,
    ctx: &mut Ctx,
    units: &[Unit],
    take: &mut dyn FnMut(&mut Ctx, FileOutcome) -> medh5::Result<()>,
) -> medh5::Result<()> {
    let kind = command(words).expect("a command that takes --workers");
    let wanted = wanted(m);
    let mut rest: Vec<usize> = (0..units.len()).collect();
    // Units that touch one file are one batch, so a pool never starts more
    // workers than there are batches: one batch is one process's work.
    let batched = if wanted > 1 && units.len() > 1 { batches(units, kind.writes) } else { Vec::new() };
    if batched.len() > 1 {
        match ctx.host.worker_command() {
            Some(program) => {
                let threads = medh5::parallel::threads()?;
                let mut pool = Pool::start(&program, words, m, wanted.min(batched.len()), threads);
                if let Some(why) = pool.failed_to_start.take().filter(|_| pool.workers.is_empty()) {
                    ctx.eprint(format!("medh5: cannot start workers ({why}); running in one process"));
                }
                let result = pool.run(units, batched, &kind, |outcome| take(ctx, outcome));
                pool.close();
                // Units the workers could not take, because none was left
                // to take them, run here.
                rest = result?;
            }
            None => ctx.eprint("medh5: this host cannot start workers; running in one process"),
        }
    }
    if rest.is_empty() {
        return Ok(());
    }
    let mut runner = (kind.runner)(m)?;
    for index in rest {
        let outcome = run_unit(runner.as_mut(), &units[index].label, &units[index].task)?;
        take(ctx, outcome)?;
    }
    Ok(())
}

/// `runner.run(task)`, but for [`test_hook`]'s `error:`.
fn run_unit(runner: &mut dyn Runner, label: &str, task: &Value) -> medh5::Result<FileOutcome> {
    if test_hook("error", label) {
        return Err(Error::Runtime(format!("{label}: stopped here, as MEDH5_TEST_WORKERS asks")));
    }
    if test_hook("threads", label) {
        let threads = medh5::parallel::threads()?;
        return Err(Error::Runtime(format!("{label}: {threads} threads, as MEDH5_TEST_WORKERS asks")));
    }
    runner.run(task)
}

/// For the test suite only: `MEDH5_TEST_WORKERS=exit:NAME` makes a worker
/// end as a crash would, without an answer, when it is handed the unit
/// named `NAME` (a file by its name, a case, a subject); `error:NAME` makes
/// that unit raise an error that stops the run, in a worker and in one
/// process alike; `threads:NAME` makes it an error that says the thread
/// count the process was given.  Nothing else can be made to do any of
/// these on demand.
fn test_hook(what: &str, label: &str) -> bool {
    let Some(hook) = std::env::var_os("MEDH5_TEST_WORKERS") else { return false };
    let hook = hook.to_string_lossy();
    matches!(hook.split_once(':'), Some((w, name)) if w == what && Path::new(label).file_name() == Some(name.as_ref()))
}

fn replay(ctx: &mut Ctx, outcome: &FileOutcome) {
    for (stream, text) in &outcome.lines {
        match stream {
            Stream::Out => ctx.print(text),
            Stream::Err => ctx.eprint(text),
        }
    }
}

// -- the worker ---------------------------------------------------------------

/// The hidden `__worker` command: `args` are the command's words and its
/// arguments, without its paths; the tasks arrive on `input`, one JSON line
/// each, and each unit's outcome --- or the error that stops it --- goes to
/// `output` as a line.  Ends at the end of `input`.
pub fn serve(args: &[String], input: &mut dyn BufRead, output: &mut dyn Write) -> i32 {
    let words: Vec<&str> = args.iter().take_while(|a| !a.starts_with('-')).map(String::as_str).collect();
    let Some(kind) = command(&words.join(" ")) else {
        let _ = writeln!(std::io::stderr(), "medh5: __worker: no command {:?} takes --workers", words.join(" "));
        return EXIT_USAGE;
    };
    let grammar = optional_paths(crate::app::command(), &words);
    let argv = std::iter::once("medh5".to_string()).chain(args.iter().cloned());
    let matches = match grammar.try_get_matches_from(argv) {
        Ok(m) => m,
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "{}", e.render());
            return EXIT_USAGE;
        }
    };
    let mut sub = &matches;
    for word in &words {
        match sub.subcommand_matches(word) {
            Some(next) => sub = next,
            None => return EXIT_USAGE,
        }
    }
    // Made at the first task: a runner that cannot be made is the error of
    // the unit that asked for it, as it would be in one process.
    let mut runner: Option<Box<dyn Runner>> = None;
    for line in input.lines() {
        let Ok(line) = line else { break };
        let Ok(task) = serde_json::from_str::<Value>(&line) else { continue };
        let (Some(index), Some(label), Some(work)) =
            (task.get("index").and_then(Value::as_u64), task.get("label").and_then(Value::as_str), task.get("task"))
        else {
            continue;
        };
        if test_hook("exit", label) {
            std::process::exit(101);
        }
        let ran = match runner.as_mut() {
            Some(r) => run_unit(r.as_mut(), label, work),
            None => (kind.runner)(sub).and_then(|made| run_unit(runner.insert(made).as_mut(), label, work)),
        };
        let answer = match ran {
            Ok(outcome) => json!({"index": index, "outcome": outcome.to_wire()}),
            Err(e) => json!({"index": index, "error": error_to_wire(&e)}),
        };
        if writeln!(output, "{}", ascii(&answer.to_string())).and_then(|()| output.flush()).is_err() {
            break;
        }
    }
    EXIT_OK
}

/// JSON `text` with every character past ASCII written as an escape.  A
/// worker's answer reaches its parent through whatever its standard output
/// encodes --- a console script's, on Windows, is the ANSI code page, which
/// writes a `§` as one byte no UTF-8 reader takes --- and ASCII is the same
/// bytes in every encoding.  Outside a string, JSON is ASCII already.
fn ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// The grammar with the paths of command `words` optional: a worker is given
/// its paths one at a time, on its standard input.
fn optional_paths(grammar: clap::Command, words: &[&str]) -> clap::Command {
    match words.split_first() {
        None if grammar.get_arguments().any(|a| a.get_id() == "paths") => {
            grammar.mut_arg("paths", |a| a.required(false))
        }
        None => grammar,
        Some((first, rest)) => grammar.mut_subcommand(*first, |sub| optional_paths(sub, rest)),
    }
}

/// The arguments of `m` as the command line gave them, but its paths and
/// `--workers`, for a worker to parse as the parent did: the options, then
/// the other positional arguments after `--`.
fn worker_args(words: &str, m: &ArgMatches) -> Vec<String> {
    let mut grammar = crate::app::command();
    let mut out: Vec<String> = words.split(' ').map(str::to_string).collect();
    for word in words.split(' ') {
        grammar = match grammar.find_subcommand(word) {
            Some(sub) => sub.clone(),
            None => return out,
        };
    }
    let mut positionals = Vec::new();
    for arg in grammar.get_arguments() {
        let id = arg.get_id().as_str();
        if id == "paths" || id == "workers" || m.value_source(id) != Some(ValueSource::CommandLine) {
            continue;
        }
        if arg.is_positional() {
            positionals.extend(m.get_raw(id).into_iter().flatten().map(|v| v.to_string_lossy().into_owned()));
            continue;
        }
        let Some(long) = arg.get_long() else { continue };
        if arg.get_action().takes_values() {
            for value in m.get_raw(id).into_iter().flatten() {
                out.push(format!("--{long}={}", value.to_string_lossy()));
            }
        } else {
            out.push(format!("--{long}"));
        }
    }
    if !positionals.is_empty() {
        out.push("--".into());
        out.extend(positionals);
    }
    out
}

/// An engine error as a line can carry it: its kind and message.
fn error_to_wire(e: &Error) -> Value {
    let kind = match e {
        Error::File(_) => "File",
        Error::Version(_) => "Version",
        Error::Schema(_) => "Schema",
        Error::Validation { .. } => "Validation",
        Error::Integrity(_) => "Integrity",
        Error::Key(_) => "Key",
        Error::Value(_) => "Value",
        Error::Type(_) => "Type",
        Error::Index(_) => "Index",
        Error::Io(_) => "Io",
        Error::NotImplemented(_) => "NotImplemented",
        Error::Runtime(_) => "Runtime",
    };
    json!({"kind": kind, "message": e.message(), "code": e.code()})
}

/// [`error_to_wire`] read back: the same error.
fn error_from_wire(value: &Value) -> Error {
    let message = value.get("message").and_then(Value::as_str).unwrap_or_default().to_string();
    match value.get("kind").and_then(Value::as_str).unwrap_or_default() {
        "File" => Error::File(message),
        "Version" => Error::Version(message),
        "Schema" => Error::Schema(message),
        "Validation" => {
            Error::Validation { message, code: value.get("code").and_then(Value::as_str).map(str::to_string) }
        }
        "Integrity" => Error::Integrity(message),
        "Key" => Error::Key(message),
        "Value" => Error::Value(message),
        "Type" => Error::Type(message),
        "Index" => Error::Index(message),
        "Io" => Error::Io(message),
        "NotImplemented" => Error::NotImplemented(message),
        _ => Error::Runtime(message),
    }
}

// -- the parent ---------------------------------------------------------------

/// Outcomes that arrive in any order, handed on in input order.
pub struct Reorder<T> {
    next: usize,
    waiting: BTreeMap<usize, T>,
}

impl<T> Reorder<T> {
    pub fn new() -> Reorder<T> {
        Reorder { next: 0, waiting: BTreeMap::new() }
    }

    /// Take the outcome of input `index`; return those now due, in order.
    pub fn push(&mut self, index: usize, item: T) -> Vec<T> {
        self.waiting.insert(index, item);
        let mut due = Vec::new();
        while let Some(item) = self.waiting.remove(&self.next) {
            due.push(item);
            self.next += 1;
        }
        due
    }
}

impl<T> Default for Reorder<T> {
    fn default() -> Self {
        Reorder::new()
    }
}

/// What a worker said: a line, or the end of its output.
type Said = (usize, Option<String>);

struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    /// The inputs it was given and has not answered, in order: the first is
    /// the one it is running.
    held: VecDeque<usize>,
    alive: bool,
}

struct Pool {
    workers: Vec<Worker>,
    said: mpsc::Receiver<Said>,
    /// Why no worker could start, when none could.
    failed_to_start: Option<String>,
}

impl Pool {
    /// `count` workers, each told this run's thread count (`MEDH5_THREADS`):
    /// `worker_args` carry only the command's own words.
    fn start(program: &[OsString], words: &str, m: &ArgMatches, count: usize, threads: usize) -> Pool {
        let (tell, said) = mpsc::channel();
        let mut workers = Vec::new();
        let mut failed_to_start = None;
        let args = worker_args(words, m);
        for _ in 0..count {
            let Some((exe, rest)) = program.split_first() else { break };
            let mut command = Command::new(exe);
            command.args(rest).arg("__worker").args(&args).env("MEDH5_THREADS", threads.to_string());
            command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
            let mut child = match medh5::h5::spawn(&mut command) {
                Ok(child) => child,
                Err(e) => {
                    failed_to_start.get_or_insert(e.to_string());
                    continue;
                }
            };
            let stdin = child.stdin.take();
            let stdout = child.stdout.take().expect("a piped stdout");
            let tell = tell.clone();
            let id = workers.len();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    match line {
                        Ok(line) => {
                            if tell.send((id, Some(line))).is_err() {
                                return;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let _ = tell.send((id, None));
            });
            workers.push(Worker { child, stdin, held: VecDeque::new(), alive: true });
        }
        Pool { workers, said, failed_to_start }
    }

    /// Run every unit, a batch to a worker; `emit` takes each outcome in
    /// input order.  Returns the
    /// units no worker was left to run, for the caller to run itself, or the
    /// error that stops the run.
    fn run(
        &mut self,
        units: &[Unit],
        batches: Vec<Vec<usize>>,
        kind: &Kind,
        mut emit: impl FnMut(FileOutcome) -> medh5::Result<()>,
    ) -> medh5::Result<Vec<usize>> {
        let mut queue: VecDeque<Vec<usize>> = batches.into();
        let mut order: Reorder<Result<FileOutcome, Error>> = Reorder::new();
        let mut done = 0;
        loop {
            for id in 0..self.workers.len() {
                let worker = &self.workers[id];
                if worker.alive && worker.stdin.is_some() && worker.held.is_empty() {
                    if let Some(batch) = queue.pop_front() {
                        self.give(id, batch, units, &mut queue);
                    }
                }
            }
            if done == units.len() {
                return Ok(Vec::new());
            }
            if !self.workers.iter().any(|w| w.alive) {
                // None left: the rest is the caller's, in input order, once
                // what has arrived is out.
                let mut rest: Vec<usize> = queue.into_iter().flatten().collect();
                rest.sort_unstable();
                return Ok(rest);
            }
            let Ok((id, line)) = self.said.recv() else { return Ok(Vec::new()) };
            let arrived = match line {
                Some(line) => match parse_answer(&line) {
                    Some((index, answer)) if self.workers[id].held.front() == Some(&index) => {
                        self.workers[id].held.pop_front();
                        vec![(index, answer)]
                    }
                    _ => continue,
                },
                None => self.died(id, units, kind, &mut queue),
            };
            for (index, answer) in arrived {
                for due in order.push(index, answer) {
                    done += 1;
                    emit(due?)?;
                }
            }
        }
    }

    /// Hand worker `id` a batch of units, one line each.
    fn give(&mut self, id: usize, batch: Vec<usize>, units: &[Unit], queue: &mut VecDeque<Vec<usize>>) {
        let worker = &mut self.workers[id];
        let mut sent = true;
        if let Some(stdin) = worker.stdin.as_mut() {
            for &index in &batch {
                let line = json!({"index": index, "label": units[index].label, "task": units[index].task});
                sent = sent && writeln!(stdin, "{line}").and_then(|()| stdin.flush()).is_ok();
            }
        } else {
            sent = false;
        }
        if sent {
            worker.held.extend(batch);
        } else {
            // A worker that cannot take input has ended, or is ending: its
            // output's end says which unit, if any, it was running.
            queue.push_front(batch);
            worker.stdin = None;
        }
    }

    /// Worker `id`'s output has ended: it is gone.  The unit it was running
    /// fails, as `kind` words it; the rest of what it held goes back on the
    /// queue, in order.
    fn died(
        &mut self,
        id: usize,
        units: &[Unit],
        kind: &Kind,
        queue: &mut VecDeque<Vec<usize>>,
    ) -> Vec<(usize, Result<FileOutcome, Error>)> {
        let worker = &mut self.workers[id];
        worker.alive = false;
        worker.stdin = None;
        let status = worker.child.wait().map(|s| s.to_string()).unwrap_or_else(|e| e.to_string());
        let mut held = std::mem::take(&mut worker.held);
        let Some(running) = held.pop_front() else { return Vec::new() };
        if !held.is_empty() {
            queue.push_front(held.into());
        }
        vec![(running, (kind.died)(&units[running], &status))]
    }

    /// End every worker: its input closed, it finishes the file it is on and
    /// exits, and is waited for.
    fn close(&mut self) {
        for worker in &mut self.workers {
            worker.stdin = None;
        }
        for worker in &mut self.workers {
            let _ = worker.child.wait();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.close();
    }
}

/// A worker's answer: the input it answers, and its outcome or error.
fn parse_answer(line: &str) -> Option<(usize, Result<FileOutcome, Error>)> {
    let value: Value = serde_json::from_str(line).ok()?;
    let index = value.get("index")?.as_u64()? as usize;
    if let Some(error) = value.get("error") {
        return Some((index, Err(error_from_wire(error))));
    }
    Some((index, Ok(FileOutcome::from_wire(value.get("outcome")?)?)))
}

/// The units in the batches workers take them: one each, but for a command
/// that writes, every unit sharing a file with another together, in input
/// order --- two workers must not write one file at once, nor one read a
/// file another is writing.
fn batches(units: &[Unit], writes: bool) -> Vec<Vec<usize>> {
    if !writes {
        return (0..units.len()).map(|i| vec![i]).collect();
    }
    // Each unit's batch, as the first unit of it: a union of the units that
    // name one file, kept by its earliest member.
    let mut first: Vec<usize> = (0..units.len()).collect();
    fn root(first: &mut [usize], mut i: usize) -> usize {
        while first[i] != i {
            first[i] = first[first[i]];
            i = first[i];
        }
        i
    }
    let mut by_file: HashMap<String, usize> = HashMap::new();
    for (index, unit) in units.iter().enumerate() {
        for path in &unit.files {
            let Some(key) = medh5::h5::file::file_key(Path::new(path)).or_else(|| lexical_key(path)) else { continue };
            match by_file.get(&key) {
                Some(&other) => {
                    let (a, b) = (root(&mut first, other), root(&mut first, index));
                    first[a.max(b)] = a.min(b);
                }
                None => {
                    by_file.insert(key, index);
                }
            }
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut at: HashMap<usize, usize> = HashMap::new();
    for index in 0..units.len() {
        let batch = root(&mut first, index);
        match at.get(&batch) {
            Some(&n) => out[n].push(index),
            None => {
                at.insert(batch, out.len());
                out.push(vec![index]);
            }
        }
    }
    out
}

/// A file that does not exist yet --- an output --- by its absolute path,
/// `.` and `..` resolved as text: two units that will write one path meet.
fn lexical_key(path: &str) -> Option<String> {
    let absolute = std::path::absolute(path).ok()?;
    let mut out = std::path::PathBuf::new();
    for part in absolute.components() {
        match part {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    // Lower case: on a file system that ignores case, two spellings are one.
    Some(format!("path:{}", out.display()).to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reorder_buffer_emits_in_input_order() {
        let mut order = Reorder::new();
        assert!(order.push(2, "c").is_empty());
        assert!(order.push(1, "b").is_empty());
        assert_eq!(order.push(0, "a"), ["a", "b", "c"]);
        assert_eq!(order.push(3, "d"), ["d"]);
        assert!(order.push(5, "f").is_empty());
        assert_eq!(order.push(4, "e"), ["e", "f"]);
    }

    #[test]
    fn an_outcome_crosses_a_line_whole() {
        let mut outcome = FileOutcome::new();
        outcome.print("a.medh5: OK\n  with a second line");
        outcome.lines.push((Stream::Err, "medh5: é".into()));
        outcome.entry = json!({"path": "a.medh5", "ok": true});
        outcome.data = json!([1, 2]);
        outcome.ok = false;
        let line = outcome.to_wire().to_string();
        assert!(!line.contains('\n'));
        let back = FileOutcome::from_wire(&serde_json::from_str(&line).unwrap()).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn an_answer_crosses_as_ascii() {
        let mut outcome = FileOutcome::new();
        outcome.print("§11.3: é, 😀");
        let line = ascii(&json!({"index": 0, "outcome": outcome.to_wire()}).to_string());
        assert!(line.is_ascii());
        let (_, back) = parse_answer(&line).unwrap();
        assert_eq!(back.unwrap(), outcome);
    }

    #[test]
    fn an_error_crosses_as_its_kind_and_message() {
        for e in [
            Error::coded("E405", "shapes differ"),
            Error::invalid("no code"),
            Error::Key("CT".into()),
            Error::File("cannot open".into()),
            Error::Io("[Errno 28] No space left on device".into()),
        ] {
            let back = error_from_wire(&error_to_wire(&e));
            assert_eq!(back.to_string(), e.to_string());
            assert_eq!(back.code(), e.code());
            assert_eq!(top_level_message(&back), top_level_message(&e));
        }
    }

    #[test]
    fn paths_naming_one_file_are_one_unit_for_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.medh5");
        let b = dir.path().join("b.medh5");
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(&b, b"b").unwrap();
        let link = dir.path().join("link.medh5");
        std::fs::hard_link(&a, &link).unwrap();
        let units: Vec<Unit> = [&a, &b, &link, &dir.path().join("missing.medh5")]
            .iter()
            .map(|p| one(&[p.display().to_string()]))
            .collect();
        assert_eq!(batches(&units, true), [vec![0, 2], vec![1], vec![3]]);
        assert_eq!(batches(&units, false), [vec![0], vec![1], vec![2], vec![3]]);
    }

    fn one(files: &[String]) -> Unit {
        Unit { task: json!(files[0]), files: files.to_vec(), label: files[0].clone() }
    }

    #[test]
    fn units_sharing_any_file_are_one_batch_for_a_writer() {
        // Two subjects writing one output (`a/../out/x` is `out/x`), and a
        // third unit reading a file of each: all one batch, in input order.
        let dir = tempfile::tempdir().unwrap();
        let at = |p: &str| dir.path().join(p).display().to_string();
        std::fs::write(dir.path().join("in1"), b"1").unwrap();
        std::fs::write(dir.path().join("in2"), b"2").unwrap();
        let units = [
            one(&[at("in1"), at("out/x.medh5")]),
            one(&[at("in2"), at("out/y.medh5")]),
            one(&[at("in3"), at("a/../out/x.medh5")]),
            one(&[at("in4"), at("out/z.medh5")]),
            one(&[at("in2"), at("out/w.medh5")]),
        ];
        assert_eq!(batches(&units, true), [vec![0, 2], vec![1, 4], vec![3]]);
        let joined = [one(&[at("in1"), at("in2")]), units[0].clone(), units[1].clone()];
        assert_eq!(batches(&joined, true), [vec![0, 1, 2]]);
    }

    #[test]
    fn a_worker_is_given_the_options_but_not_the_paths() {
        let grammar = crate::app::command();
        let m = grammar
            .try_get_matches_from([
                "medh5",
                "validate",
                "a.medh5",
                "--workers",
                "3",
                "--level",
                "strict",
                "--profile",
                "training",
                "b.medh5",
                "--json",
            ])
            .unwrap();
        let sub = m.subcommand_matches("validate").unwrap();
        assert_eq!(worker_args("validate", sub), ["validate", "--level=strict", "--profile=training", "--json"]);
        let m = crate::app::command()
            .try_get_matches_from(["medh5", "index", "build", "a.medh5", "--seed", "-3", "--workers", "2"])
            .unwrap();
        let sub = m.subcommand_matches("index").unwrap().subcommand_matches("build").unwrap();
        assert_eq!(worker_args("index build", sub), ["index", "build", "--seed=-3"]);
    }
}
