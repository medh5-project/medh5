//! Record classes: engine value types whose Python face is a frozen
//! dataclass.
//!
//! A record holds the engine value and its JSON form.  Construction builds
//! the JSON object from the arguments (positional in field order, or by
//! keyword --- the dataclass call convention) and parses it with the engine's
//! `from_json`, which is where every rule lives.  Fields read back out of the
//! JSON with the Python types 1.x returned (tuples for sequences, floats for
//! floats, nested records for nested values).
//!
//! A container field --- a dict (`Identity.extra`, `Activity.params`), or
//! nested records whose dicts are editable (`QualityRecord.agreement`) --- is
//! the record's own, as 1.x's dataclass fields were: made on its first read,
//! the same object on every read after, and read back whenever the record is
//! used (`to_json`, `==`, a writer or a document given it).  So an edit in
//! place is kept, and every rule is checked again on the record as edited.
//! A record a document hands out writes each such edit through to the
//! document as well (`medh5._live`).

use std::sync::OnceLock;

use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple};
use serde_json::{Map, Value};

use crate::convert::{json_to_py, py_to_json};
use crate::errors::R;

/// How one field reads back.
#[derive(Clone, Copy)]
pub enum Kind {
    /// The JSON value as is (`None` when absent).
    Plain,
    /// A JSON number as `float`.
    Float,
    /// A JSON array as a `tuple` (`()` when absent).
    Tuple,
    /// A JSON array of pairs as a tuple of tuples.
    Pairs,
    /// A JSON object as a `dict` (`{}` when absent).
    Dict,
    /// A string with a default when absent.
    Default(&'static str),
    /// `False` when absent.
    FalseDefault,
    /// A nested record, built by the given constructor (`None` when absent).
    Record(MakeRecord),
    /// A tuple of nested records.
    Records(MakeRecord),
    /// Every top-level member not named by another field (`{}` when none).
    Extra,
}

/// A nested record's constructor: its JSON form, and the anchor its
/// containers write through, if it has one.
pub type MakeRecord = fn(Python<'_>, &Value, Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>>;

impl Kind {
    /// Whether the field holds something an edit in place can change: a
    /// dict, or nested records, which have dicts of their own.
    pub fn is_container(self) -> bool {
        matches!(self, Kind::Dict | Kind::Extra | Kind::Record(_) | Kind::Records(_))
    }
}

pub struct Field {
    pub name: &'static str,
    pub key: &'static str,
    pub kind: Kind,
    /// A dataclass field without a default: the constructor needs it.
    pub required: bool,
    /// What reading the field warns, when 3.0 removes it.
    pub deprecated: Option<&'static str>,
}

pub const fn field(name: &'static str, kind: Kind) -> Field {
    Field { name, key: name, kind, required: false, deprecated: None }
}

/// A field the constructor cannot do without.
pub const fn required(name: &'static str, kind: Kind) -> Field {
    Field { name, key: name, kind, required: true, deprecated: None }
}

impl Field {
    /// The field, warning `message` (which starts "medh5 3.0 ") when read.
    pub const fn deprecated(self, message: &'static str) -> Field {
        Field { deprecated: Some(message), ..self }
    }
}

/// The constructor's `inspect.Signature`: each field a parameter, positional
/// or keyword, an optional one defaulting to what it reads as when absent.
/// The constructor itself takes `*args, **kwargs` and matches them to the
/// fields, so without it `help()` and `inspect` would name no parameter.
pub fn signature(py: Python<'_>, fields: &[Field]) -> PyResult<Py<PyAny>> {
    let inspect = py.import("inspect")?;
    let parameter = inspect.getattr("Parameter")?;
    let kind = parameter.getattr("POSITIONAL_OR_KEYWORD")?;
    let parameters = fields
        .iter()
        .map(|f| {
            let kwargs = PyDict::new(py);
            if !f.required {
                let default = match f.kind {
                    Kind::Tuple | Kind::Pairs | Kind::Records(_) => PyTuple::empty(py).into_any(),
                    Kind::Dict | Kind::Extra => PyDict::new(py).into_any(),
                    Kind::Default(text) => text.into_pyobject(py)?.into_any(),
                    Kind::FalseDefault => false.into_pyobject(py)?.to_owned().into_any(),
                    Kind::Plain | Kind::Float | Kind::Record(_) => py.None().into_bound(py),
                };
                kwargs.set_item("default", default)?;
            }
            parameter.call((f.name, &kind), Some(&kwargs))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(inspect.getattr("Signature")?.call1((parameters,))?.unbind())
}

/// `missing 2 required positional arguments: 'a' and 'b'`, as a dataclass says it.
fn missing_arguments(class: &str, missing: &[&str]) -> PyErr {
    let quoted: Vec<String> = missing.iter().map(|n| medh5::json::repr_str(n)).collect();
    let names = match quoted.as_slice() {
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
        [] => String::new(),
    };
    let plural = if missing.len() == 1 { "argument" } else { "arguments" };
    PyTypeError::new_err(format!("{class}.__init__() missing {} required positional {plural}: {names}", missing.len()))
}

/// Read one field from a record's JSON form.  A container field is made
/// new: anchored, when `anchor` is given, so that an edit to it is written
/// back through the anchor (`medh5._live`).
pub fn read_field<'py>(
    py: Python<'py>,
    json: &Value,
    fields: &[Field],
    f: &Field,
    anchor: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let value = json.get(f.key);
    let none = || py.None().into_bound(py);
    Ok(match f.kind {
        Kind::Plain => match value {
            Some(v) => json_to_py(py, v)?,
            None => none(),
        },
        Kind::Float => match value.and_then(Value::as_f64) {
            Some(v) => v.into_pyobject(py)?.into_any(),
            None => match value {
                Some(Value::Null) | None => none(),
                Some(other) => json_to_py(py, other)?,
            },
        },
        Kind::Tuple => match value {
            Some(Value::Array(items)) => {
                PyTuple::new(py, items.iter().map(|i| json_to_py(py, i)).collect::<PyResult<Vec<_>>>()?)?.into_any()
            }
            Some(Value::Null) | None => PyTuple::empty(py).into_any(),
            Some(other) => json_to_py(py, other)?,
        },
        Kind::Pairs => match value {
            Some(Value::Array(items)) => {
                let pairs = items
                    .iter()
                    .map(|i| match i {
                        Value::Array(inner) => Ok(PyTuple::new(
                            py,
                            inner.iter().map(|v| json_to_py(py, v)).collect::<PyResult<Vec<_>>>()?,
                        )?
                        .into_any()),
                        other => json_to_py(py, other),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, pairs)?.into_any()
            }
            _ => PyTuple::empty(py).into_any(),
        },
        Kind::Dict => {
            let plain = match value {
                Some(v @ Value::Object(_)) => json_to_py(py, v)?,
                _ => PyDict::new(py).into_any(),
            };
            containers(plain, anchor)?
        }
        Kind::Default(text) => match value {
            Some(v) if !v.is_null() => json_to_py(py, v)?,
            _ => text.into_pyobject(py)?.into_any(),
        },
        Kind::FalseDefault => match value {
            Some(v) if !v.is_null() => json_to_py(py, v)?,
            _ => false.into_pyobject(py)?.to_owned().into_any(),
        },
        Kind::Record(make) => match value {
            Some(v) if !v.is_null() => make(py, v, anchor)?.into_bound(py),
            _ => none(),
        },
        Kind::Records(make) => match value {
            Some(Value::Array(items)) => {
                PyTuple::new(py, items.iter().map(|i| make(py, i, anchor)).collect::<PyResult<Vec<_>>>()?)?.into_any()
            }
            _ => PyTuple::empty(py).into_any(),
        },
        Kind::Extra => {
            let out = PyDict::new(py);
            if let Value::Object(map) = json {
                for (k, v) in map {
                    if !fields.iter().any(|f| f.key == k) {
                        out.set_item(k, json_to_py(py, v)?)?;
                    }
                }
            }
            containers(out.into_any(), anchor)?
        }
    })
}

/// `plain` (a JSON value), or with an anchor, its objects and arrays as
/// containers that write an edit back through it.
pub fn containers<'py>(plain: Bound<'py, PyAny>, anchor: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    static NESTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    match anchor {
        Some(anchor) => NESTED.import(plain.py(), "medh5._live", "_nested")?.call1((plain, anchor)),
        None => Ok(plain),
    }
}

/// A new `medh5._live._Anchor`: for the record read as `token` from member
/// `field` (and `key`) of `document`, whose `_replace_member` it calls.
pub fn new_anchor<'py>(
    document: &Bound<'py, PyAny>,
    field: &str,
    key: Option<&str>,
    token: &str,
) -> PyResult<Bound<'py, PyAny>> {
    static ANCHOR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    ANCHOR.import(document.py(), "medh5._live", "_Anchor")?.call1((document, field, key, token))
}

/// Where a record a document handed out was read from.
struct Member {
    document: Py<PyAny>,
    field: String,
    key: Option<String>,
}

/// What a record keeps beside its JSON form: its container fields, once
/// read, and where it was read from, if a document handed it out.
pub struct Held {
    /// One cell per field, filled for a container field on its first read.
    cells: Box<[OnceLock<Py<PyAny>>]>,
    /// The document member the record was read from.
    member: Option<Member>,
    /// The anchor its containers write through: made when the first of them
    /// is read, so that a record only read for its other fields is never
    /// anchored.
    anchor: OnceLock<Py<PyAny>>,
}

impl Held {
    pub fn new(fields: usize) -> Self {
        Held { cells: (0..fields).map(|_| OnceLock::new()).collect(), member: None, anchor: OnceLock::new() }
    }

    /// Whether no container has been read: the JSON form is then the record.
    pub fn is_untouched(&self) -> bool {
        self.cells.iter().all(|c| c.get().is_none())
    }

    /// Whether the containers write through an anchor (`medh5._live`), and
    /// so hold only what was converted for them.  A record no document
    /// handed out holds the caller's own dicts, which no lock guards.
    pub fn is_anchored(&self) -> bool {
        self.anchor.get().is_some()
    }

    /// Field `i`'s container, if it has been read.
    pub fn get(&self, i: usize) -> Option<&Py<PyAny>> {
        self.cells[i].get()
    }

    /// Keep `value` as field `i`'s container, unless one already is (a read
    /// on another thread got there first); the one kept.
    pub fn keep<'py>(&self, i: usize, value: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = value.py();
        let _ = self.cells[i].set(value.clone().unbind());
        self.cells[i].get().map(|v| v.bind(py).clone()).unwrap_or(value)
    }

    /// Record that the record was read from member `field` (`key`) of
    /// `document`.
    pub fn read_from(&mut self, document: Py<PyAny>, field: &str, key: Option<&str>) {
        self.member = Some(Member { document, field: field.to_string(), key: key.map(str::to_string) });
    }

    /// The anchor the containers write through: given, made now from the
    /// member read (with `record` as its root), or none for a record no
    /// document handed out.
    pub fn anchor<'py>(
        &self,
        record: &Bound<'py, PyAny>,
        json: &Value,
        given: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = record.py();
        if let Some(found) = self.anchor.get() {
            return Ok(Some(found.bind(py).clone()));
        }
        let made = match (given, &self.member) {
            (Some(anchor), _) => anchor.clone(),
            (None, Some(m)) => {
                let token = medh5::json::canonical(json);
                let anchor = new_anchor(m.document.bind(py), &m.field, m.key.as_deref(), &token)?;
                anchor.setattr("root", record)?;
                anchor
            }
            (None, None) => return Ok(None),
        };
        let _ = self.anchor.set(made.unbind());
        Ok(self.anchor.get().map(|a| a.bind(py).clone()))
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for cell in self.cells.iter() {
            visit.call(cell.get())?;
        }
        if let Some(m) = &self.member {
            visit.call(&m.document)?;
        }
        visit.call(self.anchor.get())
    }
}

/// A record's JSON form with each container read put back: a dict as it
/// now stands (omitted when empty), the extra members replaced by the
/// `extra` dict's, nested records as each now stands.  Not validated: the
/// caller parses it.
pub fn merged_json(py: Python<'_>, class: &str, json: &Value, fields: &[Field], held: &Held) -> R<Value> {
    let mut out = match json {
        Value::Object(m) => m.clone(),
        _ => Map::new(),
    };
    let named = |k: &str| fields.iter().any(|f| f.key == k);
    for (i, f) in fields.iter().enumerate() {
        let Some(value) = held.get(i) else { continue };
        let value = value.bind(py);
        match f.kind {
            Kind::Dict => match py_to_json(value)? {
                Value::Object(m) if m.is_empty() => {
                    out.shift_remove(f.key);
                }
                v => {
                    out.insert(f.key.to_string(), v);
                }
            },
            Kind::Extra => {
                out.retain(|k, _| named(k));
                if let Value::Object(extra) = py_to_json(value)? {
                    for (k, v) in extra {
                        if named(&k) {
                            return Err(medh5::Error::invalid(format!(
                                "{class}.{} holds {}, which is a field of {class}, not an extra member; \
                                 give the field itself (dataclasses.replace)",
                                f.name,
                                medh5::json::repr_str(&k)
                            ))
                            .into());
                        }
                        out.insert(k, v);
                    }
                }
            }
            Kind::Record(_) => {
                if value.is_none() {
                    out.shift_remove(f.key);
                } else {
                    out.insert(f.key.to_string(), py_to_json(&value.call_method0("to_json")?)?);
                }
            }
            Kind::Records(_) => {
                let items = value
                    .try_iter()?
                    .map(|r| py_to_json(&r?.call_method0("to_json")?))
                    .collect::<PyResult<Vec<_>>>()?;
                if items.is_empty() {
                    out.shift_remove(f.key);
                } else {
                    out.insert(f.key.to_string(), Value::Array(items));
                }
            }
            _ => {}
        }
    }
    Ok(Value::Object(out))
}

/// A hash of the fields an edit cannot change.  A container can be edited
/// in place, and a hash must not change while the record lives; records
/// equal as they stand agree on these fields, so they hash alike.
pub fn fixed_hash(json: &Value, fields: &[Field]) -> isize {
    let mut fixed = Map::new();
    for f in fields.iter().filter(|f| !f.kind.is_container()) {
        if let Some(v) = json.get(f.key) {
            fixed.insert(f.key.to_string(), v.clone());
        }
    }
    crate::values::json_hash(&Value::Object(fixed))
}

/// The JSON object a constructor call describes: arguments matched to
/// fields positionally, then by keyword; `None` and empty values omitted.
pub fn build_json(
    class: &str,
    fields: &[Field],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Value> {
    if args.len() > fields.len() {
        // As a dataclass counts them: `self`, then the required fields.
        let required = fields.iter().filter(|f| f.required).count() + 1;
        let most = fields.len() + 1;
        let takes = match (required == most, most) {
            (true, 1) => "1 positional argument".to_string(),
            (true, _) => format!("{most} positional arguments"),
            (false, _) => format!("from {required} to {most} positional arguments"),
        };
        return Err(PyTypeError::new_err(format!(
            "{class}.__init__() takes {takes} but {} were given",
            args.len() + 1
        )));
    }
    let mut given: Vec<Option<Bound<'_, PyAny>>> = vec![None; fields.len()];
    for (i, arg) in args.iter().enumerate() {
        given[i] = Some(arg);
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let name: String = k.extract()?;
            let Some(i) = fields.iter().position(|f| f.name == name) else {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got an unexpected keyword argument {}",
                    medh5::json::repr_str(&name)
                )));
            };
            if given[i].is_some() {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got multiple values for argument {}",
                    medh5::json::repr_str(&name)
                )));
            }
            given[i] = Some(v);
        }
    }
    let missing: Vec<&str> =
        fields.iter().zip(&given).filter(|(f, v)| f.required && v.is_none()).map(|(f, _)| f.name).collect();
    if !missing.is_empty() {
        return Err(missing_arguments(class, &missing));
    }
    let mut out = Map::new();
    for (f, value) in fields.iter().zip(given) {
        let Some(value) = value else { continue };
        if value.is_none() {
            continue;
        }
        let json = py_to_json(&value)?;
        match f.kind {
            Kind::Extra => {
                if let Value::Object(extra) = json {
                    for (k, v) in extra {
                        out.entry(k).or_insert(v);
                    }
                }
            }
            Kind::Tuple | Kind::Pairs | Kind::Records(_) if json.as_array().is_some_and(Vec::is_empty) => {}
            Kind::Dict if json.as_object().is_some_and(Map::is_empty) => {}
            _ => {
                out.insert(f.key.to_string(), json);
            }
        }
    }
    Ok(Value::Object(out))
}

pub fn missing_attribute(class: &str, name: &str) -> PyErr {
    PyAttributeError::new_err(format!(
        "{} object has no attribute {}",
        medh5::json::repr_str(class),
        medh5::json::repr_str(name)
    ))
}

/// Define a record class over an engine value type.
///
/// `parse` turns the JSON form into the engine value (validating it);
/// `dump` turns the engine value back into JSON.  Doc comments before the
/// struct's name become the class's docstring.
#[macro_export]
macro_rules! record_class {
    (@nullable) => { false };
    (@nullable $flag:expr) => { $flag };
    (
        $(#[$meta:meta])*
        $rust:ident, $py:literal, $module:literal, $engine:ty,
        parse = $parse:expr,
        dump = $dump:expr,
        $(nullable = $nullable:expr,)?
        fields = [$($field:expr),* $(,)?]
        $(, methods = { $($methods:tt)* })?
    ) => {
        $(#[$meta])*
        #[pyclass(module = $module, name = $py, skip_from_py_object, frozen)]
        pub struct $rust {
            /// The record as made.  Its container fields may have been
            /// edited since: `current` is the record as it stands.
            pub inner: $engine,
            /// `inner`'s JSON form.
            pub json: serde_json::Value,
            held: $crate::records::Held,
        }

        impl $rust {
            pub const FIELDS: &'static [$crate::records::Field] = &[$($field),*];
            /// `from_json` of an absent or empty document is `None`.
            pub const NULLABLE: bool = $crate::record_class!(@nullable $($nullable)?);

            pub fn wrap(inner: $engine) -> Self {
                let dump: fn(&$engine) -> serde_json::Value = $dump;
                let json = dump(&inner);
                $rust { inner, json, held: $crate::records::Held::new(Self::FIELDS.len()) }
            }

            /// The record of member `field` (`key`) of `document`, read as
            /// `inner`: an edit to one of its containers is written back to
            /// the member, while the member is still what was read.
            pub fn read_from(inner: $engine, document: &Bound<'_, PyAny>, field: &str, key: Option<&str>) -> Self {
                let mut made = Self::wrap(inner);
                made.held.read_from(document.clone().unbind(), field, key);
                made
            }

            pub fn from_value(value: &serde_json::Value) -> $crate::errors::R<Self> {
                let parse: fn(&serde_json::Value) -> medh5::Result<$engine> = $parse;
                Ok(Self::wrap(parse(value)?))
            }

            /// A Python object for a nested JSON value (used by `Kind::Record`),
            /// its containers anchored to `anchor` if one is given.
            pub fn py_from_json(
                py: Python<'_>,
                value: &serde_json::Value,
                anchor: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let made = Bound::new(py, Self::from_value(value).map_err(PyErr::from)?)?;
                if let Some(anchor) = anchor {
                    made.get().held.anchor(made.as_any(), &made.get().json, Some(anchor))?;
                    Self::anchor_containers(&made, anchor)?;
                }
                Ok(made.into_any().unbind())
            }

            /// The record as it stands --- its containers as edited --- parsed
            /// again, so that every rule holds of it; and its JSON form.
            pub fn current(&self, py: Python<'_>) -> $crate::errors::R<($engine, serde_json::Value)> {
                if self.held.is_untouched() {
                    return Ok((self.inner.clone(), self.json.clone()));
                }
                // Under the lock an edit to an anchored container holds with
                // its rewrite (`medh5._live`): one state of every container.
                // A record no document handed out is read without it: its
                // containers are the caller's, holding the caller's objects,
                // whose conversion is the caller's code.
                let merge = || $crate::records::merged_json(py, $py, &self.json, Self::FIELDS, &self.held);
                let merged = if self.held.is_anchored() {
                    $crate::document::under_live_lock(py, merge)?
                } else {
                    merge()?
                };
                let parse: fn(&serde_json::Value) -> medh5::Result<$engine> = $parse;
                let dump: fn(&$engine) -> serde_json::Value = $dump;
                let inner = parse(&merged)?;
                let json = dump(&inner);
                Ok((inner, json))
            }

            /// The engine value of a Python argument, and its JSON form: this
            /// class as it stands, or its JSON form.
            pub fn current_of(obj: &Bound<'_, PyAny>) -> $crate::errors::R<($engine, serde_json::Value)> {
                if let Ok(rec) = obj.cast::<$rust>() {
                    return rec.get().current(obj.py());
                }
                let parse: fn(&serde_json::Value) -> medh5::Result<$engine> = $parse;
                let dump: fn(&$engine) -> serde_json::Value = $dump;
                let inner = parse(&$crate::convert::py_to_json(obj)?)?;
                let json = dump(&inner);
                Ok((inner, json))
            }

            /// The engine value of a Python argument: this class as it stands,
            /// or its JSON form.
            pub fn arg(obj: &Bound<'_, PyAny>) -> $crate::errors::R<$engine> {
                Ok(Self::current_of(obj)?.0)
            }

            /// Make every container of `record`, anchored to `anchor`.
            fn anchor_containers(record: &Bound<'_, Self>, anchor: &Bound<'_, PyAny>) -> PyResult<()> {
                let py = record.py();
                let me = record.get();
                for (i, f) in Self::FIELDS.iter().enumerate() {
                    if f.kind.is_container() {
                        me.held.keep(i, $crate::records::read_field(py, &me.json, Self::FIELDS, f, Some(anchor))?);
                    }
                }
                Ok(())
            }

            /// Field `i`, read: a container is made on its first read --- with
            /// every other container, anchored, for a record a document
            /// handed out --- and kept.
            fn field_value<'py>(record: &Bound<'py, Self>, i: usize) -> PyResult<Bound<'py, PyAny>> {
                let py = record.py();
                let me = record.get();
                let f = &Self::FIELDS[i];
                if !f.kind.is_container() {
                    return $crate::records::read_field(py, &me.json, Self::FIELDS, f, None);
                }
                if let Some(found) = me.held.get(i) {
                    return Ok(found.bind(py).clone());
                }
                if let Some(anchor) = me.held.anchor(record.as_any(), &me.json, None)? {
                    Self::anchor_containers(record, &anchor)?;
                    if let Some(found) = me.held.get(i) {
                        return Ok(found.bind(py).clone());
                    }
                }
                Ok(me.held.keep(i, $crate::records::read_field(py, &me.json, Self::FIELDS, f, None)?))
            }
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (*args, **kwargs))]
            fn __new__(args: &Bound<'_, pyo3::types::PyTuple>, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> $crate::errors::R<Self> {
                let json = $crate::records::build_json($py, Self::FIELDS, args, kwargs)?;
                Self::from_value(&json)
            }

            #[classattr]
            fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
                let names: Vec<&str> = Self::FIELDS.iter().map(|f| f.name).collect();
                $crate::geometry::dataclass_fields(py, &names)
            }

            #[classattr]
            fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
                Ok(pyo3::types::PyTuple::new(py, Self::FIELDS.iter().map(|f| f.name))?.unbind())
            }

            #[classattr]
            fn __signature__(py: Python<'_>) -> PyResult<Py<PyAny>> {
                $crate::records::signature(py, Self::FIELDS)
            }

            fn __getattr__<'py>(slf: &Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
                match Self::FIELDS.iter().position(|f| f.name == name) {
                    Some(i) => {
                        if let Some(message) = Self::FIELDS[i].deprecated {
                            $crate::deprecation::warn(slf.py(), message, false)?;
                        }
                        Self::field_value(slf, i)
                    }
                    None => Err($crate::records::missing_attribute($py, name)),
                }
            }

            /// The record as it stands, as JSON.
            fn to_json<'py>(&self, py: Python<'py>) -> $crate::errors::R<Bound<'py, PyAny>> {
                Ok($crate::convert::json_to_py(py, &self.current(py)?.1)?)
            }

            /// The fields too: they are read through `__getattr__`, and
            /// completion in a notebook goes by `dir()`.
            fn __dir__(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
                let object = slf.py().import("builtins")?.getattr("object")?;
                let mut names: Vec<String> = object.getattr("__dir__")?.call1((slf,))?.extract()?;
                names.extend(Self::FIELDS.iter().map(|f| f.name.to_string()));
                names.sort();
                names.dedup();
                Ok(names)
            }

            /// `copy.replace(record, **changes)`.
            #[pyo3(signature = (**changes))]
            fn __replace__(
                slf: &Bound<'_, Self>,
                changes: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> PyResult<Py<PyAny>> {
                // A replacement reads every field; a deprecated one warns only
                // when the caller reads it.
                if Self::FIELDS.iter().any(|f| f.deprecated.is_some()) {
                    return $crate::values::quietly(slf.py(), || $crate::values::dataclass_replace(slf.as_any(), changes));
                }
                $crate::values::dataclass_replace(slf.as_any(), changes)
            }

            #[classmethod]
            fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> $crate::errors::R<Option<Self>> {
                if Self::NULLABLE && (doc.is_none() || !doc.is_truthy()?) {
                    return Ok(None);
                }
                Self::from_value(&$crate::convert::py_to_json(doc)?).map(Some)
            }

            /// Equal as they stand; a record whose edits break a rule equals
            /// none.
            fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
                let Ok(other) = other.cast::<$rust>() else { return false };
                match (self.current(py), other.get().current(py)) {
                    (Ok(a), Ok(b)) => a.1 == b.1,
                    _ => false,
                }
            }

            fn __hash__(&self) -> isize {
                $crate::records::fixed_hash(&self.json, Self::FIELDS)
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                let me = slf.get();
                let fields = Self::FIELDS
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let value = match me.held.get(i) {
                            Some(found) => found.bind(py).clone(),
                            None => $crate::records::read_field(py, &me.json, Self::FIELDS, f, None)?,
                        };
                        Ok((f.name, value))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                $crate::values::dataclass_repr($py, &fields)
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
                $crate::values::reduce_via_json(slf.as_any())
            }

            /// Anchor a record just read to `anchor` (`medh5._live`): its
            /// containers, made now, write an edit back through it.
            fn _anchor_to(slf: &Bound<'_, Self>, anchor: &Bound<'_, PyAny>) -> PyResult<()> {
                let me = slf.get();
                if !me.held.is_untouched() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "a record whose fields have been read cannot be anchored",
                    ));
                }
                me.held.anchor(slf.as_any(), &me.json, Some(anchor))?;
                Self::anchor_containers(slf, anchor)
            }

            fn __traverse__(&self, visit: pyo3::pyclass::PyVisit<'_>) -> Result<(), pyo3::pyclass::PyTraverseError> {
                self.held.traverse(&visit)
            }

            $($($methods)*)?
        }
    };
}
