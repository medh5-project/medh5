//! Bundled vocabularies and the vocabulary registry (spec §5.1).
//!
//! Three vocabularies ship with the engine, chosen because they cover the shapes
//! a label set can take rather than because they are exhaustive: one class
//! (`binary-foreground`), a small hierarchy with overlap-capable sub-regions
//! (`brats-subregions`), and a flat multi-organ set (`amos22-organs`).
//!
//! `brats-subregions` uses the BraTS 2023 numbering (enhancing tumour is 3) and
//! adds a synthetic parent, `tumor`, as id 4.  BraTS releases up to 2021 label
//! enhancing tumour 4: their masks are relabelled 4 to 3 before they are
//! written with this vocabulary, or their enhancing tumour lands on the parent.
//!
//! **No ontology codes are bundled.**  A wrong SNOMED-CT or FMA binding is a
//! silent data-integrity defect that propagates into every file written with
//! the vocabulary and that no validator can detect.  Bindings are the curator's
//! to add, and the validator's W912 says so when they are missing.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock, PoisonError};

use serde_json::{json, Value};

use super::{LabelClass, LabelSet, Relation, Skeleton};
use crate::json::{repr_list, repr_str};
use crate::pyval::{self, get, get_list, require, to_str};
use crate::{Error, Result};

const BUNDLED: [(&str, &str); 3] = [
    ("amos22-organs", include_str!("../../data/vocabularies/amos22-organs.json")),
    ("binary-foreground", include_str!("../../data/vocabularies/binary-foreground.json")),
    ("brats-subregions", include_str!("../../data/vocabularies/brats-subregions.json")),
];

fn extra() -> &'static Mutex<BTreeMap<String, LabelSet>> {
    static EXTRA: OnceLock<Mutex<BTreeMap<String, LabelSet>>> = OnceLock::new();
    EXTRA.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// The bundled vocabulary documents, by name.
pub fn bundled() -> impl Iterator<Item = (&'static str, &'static str)> {
    BUNDLED.iter().copied()
}

/// Every vocabulary name [`load`] accepts, bundled and registered, sorted.
pub fn available() -> Vec<String> {
    let mut names: Vec<String> = BUNDLED.iter().map(|(n, _)| n.to_string()).collect();
    names.extend(extra().lock().unwrap_or_else(PoisonError::into_inner).keys().cloned());
    names.sort();
    names.dedup();
    names
}

/// A label set from a vocabulary document (`form` and `sha256` ignored).
pub fn labelset_from_doc(doc: &Value) -> Result<LabelSet> {
    let doc = pyval::as_object(doc, "a vocabulary")?;
    let classes = get_list(doc, "classes").iter().map(LabelClass::from_json).collect::<Result<Vec<_>>>()?;
    let relations = get_list(doc, "relations").iter().map(Relation::from_json).collect::<Result<Vec<_>>>()?;
    let skeletons = get_list(doc, "skeletons").iter().map(Skeleton::from_json).collect::<Result<Vec<_>>>()?;
    LabelSet::new(
        to_str(require(doc, "id")?),
        classes,
        get(doc, "version").map(to_str).unwrap_or_else(|| "1.0.0".into()),
        relations,
        skeletons,
        "inline",
        None,
        None,
    )
}

/// Load a bundled or registered vocabulary by name.
pub fn load(name: &str) -> Result<LabelSet> {
    if let Some(found) = extra().lock().unwrap_or_else(PoisonError::into_inner).get(name) {
        return Ok(found.clone());
    }
    let Some((_, text)) = BUNDLED.iter().find(|(n, _)| *n == name) else {
        return Err(Error::coded(
            "E305",
            format!("unknown vocabulary {}; available: {}", repr_str(name), repr_list(&available())),
        ));
    };
    labelset_from_doc(&serde_json::from_str(text)?)
}

/// Load a vocabulary from a JSON file on disk.
pub fn load_file(path: &Path) -> Result<LabelSet> {
    let text = std::fs::read_to_string(path)?;
    labelset_from_doc(&serde_json::from_str(&text)?)
}

/// Make a vocabulary loadable by name for the rest of the process.
pub fn register(name: &str, label_set: LabelSet) -> LabelSet {
    extra().lock().unwrap_or_else(PoisonError::into_inner).insert(name.to_string(), label_set.clone());
    label_set
}

/// Drop a registered vocabulary; bundled ones cannot be removed.
pub fn unregister(name: &str) {
    extra().lock().unwrap_or_else(PoisonError::into_inner).remove(name);
}

/// Name -> `{id, version, classes, sha256}`, for `medh5 labels registry list`.
///
/// The registered vocabularies are taken in one step, so a name another
/// thread unregisters meanwhile is either described or absent, never listed
/// and then looked up in vain (E305).
pub fn describe() -> Result<serde_json::Map<String, Value>> {
    let registered = extra().lock().unwrap_or_else(PoisonError::into_inner).clone();
    let mut sets = BTreeMap::new();
    for (name, text) in BUNDLED {
        if !registered.contains_key(name) {
            sets.insert(name.to_string(), labelset_from_doc(&serde_json::from_str(text)?)?);
        }
    }
    sets.extend(registered);
    let mut out = serde_json::Map::new();
    for (name, ls) in sets {
        out.insert(name, json!({"id": ls.id, "version": ls.version, "classes": ls.len(), "sha256": ls.sha256()}));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_vocabularies_load() {
        assert_eq!(available(), vec!["amos22-organs", "binary-foreground", "brats-subregions"]);
        for name in available() {
            let ls = load(&name).unwrap();
            assert!(!ls.is_empty());
        }
        assert_eq!(load("nope").unwrap_err().code(), Some("E305"));
    }
}
