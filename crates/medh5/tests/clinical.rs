//! The clinical profile end to end (format 1.1): what amendment, packing,
//! recompression and selection promise, through the public API.

use std::path::{Path, PathBuf};

use medh5::clinical::augment::{augment, imaging_events_from_timepoints, strip};
use medh5::clinical::model::{Bounds, ClinicalRecords, Clock, Descriptor, Event, Link, HOUR};
use medh5::clinical::{select, SelectionPolicy};
use medh5::collection::{extract, open_collection, pack};
use medh5::conformance::clinical::{worked_example, worked_records, REPORT_V1};
use medh5::integrity::collect_digests;
use medh5::sample::{amend, create, open_sample, GridOptions, ImageOptions};
use medh5::storage::recompress::recompress;
use medh5::validate::validate_file;
use medh5::Error;

fn worked(dir: &Path) -> PathBuf {
    let path = dir.join("worked.medh5");
    worked_example(&path).unwrap();
    path
}

fn imaging_only(path: &Path) {
    let ct = medh5::array::NdArray::from_vec(&[8, 16, 16], vec![0i16; 8 * 16 * 16]).unwrap();
    let mut w = create(path, Some("img"), Some("SUBJ-9"), "balanced", &[]).unwrap();
    w.add_grid("ct", &[8, 16, 16], &[2.0, 0.8, 0.8], GridOptions::default()).unwrap();
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default()).unwrap();
    w.commit(true).unwrap();
}

fn content_id(path: &Path) -> String {
    open_sample(path).unwrap().content_id().unwrap().unwrap()
}

fn payload_digests(path: &Path) -> Vec<(String, String)> {
    let s = open_sample(path).unwrap();
    let mut d: Vec<(String, String)> = collect_digests(&s.root, &["index", "clinical"]).unwrap().into_iter().collect();
    d.sort();
    d
}

#[test]
fn s2_ordinary_imaging_writes_stay_1_0() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.medh5");
    imaging_only(&path);
    let s = open_sample(&path).unwrap();
    assert_eq!(s.version().unwrap(), "1.0");
    assert!(!s.profiles().unwrap().contains("clinical"));
    assert!(s.clinical().unwrap().is_none());
}

#[test]
fn s8_a_no_op_amend_preserves_every_byte_of_the_profile() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let before = content_id(&path);
    let mut w = amend(&path, None).unwrap();
    assert!(w.has_clinical());
    w.commit(true).unwrap();
    assert_eq!(content_id(&path), before);
    let s = open_sample(&path).unwrap();
    assert_eq!(s.version().unwrap(), "1.1");
    assert!(s.profiles().unwrap().contains("clinical"));
    assert_eq!(s.clinical().unwrap().unwrap().text("report0_text_v1").unwrap(), REPORT_V1);
}

#[test]
fn s8_rewriting_unchanged_records_reproduces_the_content_id() {
    // Loading the records forces a rewrite: storage order and encoding are a
    // function of the records, so the bytes --- and the address --- come back.
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let before = content_id(&path);
    let mut w = amend(&path, None).unwrap();
    w.clinical().unwrap().unwrap();
    w.commit(true).unwrap();
    assert_eq!(content_id(&path), before);
}

#[test]
fn s8_adding_an_event_changes_the_address_and_no_payload_digest() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let (before, payloads) = (content_id(&path), payload_digests(&path));
    let mut w = amend(&path, None).unwrap();
    w.add_event(Event {
        event_id: "lab1".into(),
        record_id: "lab1".into(),
        kind: "observation".into(),
        temporal_type: "point".into(),
        effective_start: Some(Bounds::exact(100 * HOUR)),
        available: Some(Bounds::exact(101 * HOUR)),
        status: "final".into(),
        code_system: Some("http://loinc.org".into()),
        code: Some("2160-0".into()),
        value_num: Some(1.3),
        unit: Some("mg/dL".into()),
        ..Default::default()
    })
    .unwrap();
    w.commit(true).unwrap();
    assert_ne!(content_id(&path), before);
    assert_eq!(payload_digests(&path), payloads);
    assert!(validate_file(&path, "integrity", None).unwrap().ok());
}

#[test]
fn s5_an_event_version_is_immutable() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let mut w = amend(&path, None).unwrap();
    let mut copy = worked_records().events[0].clone();
    copy.value_num = Some(9.9);
    let err = w.add_event(copy).unwrap_err();
    assert_eq!(err.code(), Some("E809"));
    w.abort();
}

#[test]
fn s7_3_an_amendment_cannot_change_what_a_held_version_attests() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let before = content_id(&path);
    // A grounding drawn now, asserted by the report it grounds: refused as it
    // is added.
    let mut w = amend(&path, None).unwrap();
    let mut grounding = Link::new(("event", "report0_v1"), "describes", ("annotation", "lesions_tp0"));
    grounding.asserted_by_event_id = Some("report0_v1".into());
    assert_eq!(w.add_link(grounding).unwrap_err().code(), Some("E809"));
    w.abort();
    // A mask the procedure `seg0` attests, removed: refused at the commit,
    // which compares what is attested with the file being amended.
    let mut w = amend(&path, None).unwrap();
    w.remove_annotation("lesions_tp1").unwrap();
    let err = w.commit(true).unwrap_err();
    assert_eq!(err.code(), Some("E809"), "{err}");
    assert!(err.to_string().contains("annotation 'lesions_tp1', which event 'seg0' attests"), "{err}");
    // The history dropped and added again, with a second text for the
    // preliminary report: held to what it was.
    let mut w = amend(&path, None).unwrap();
    w.drop_clinical().unwrap();
    w.add_records(worked_records()).unwrap();
    w.add_document(medh5::clinical::model::Document::text("late", "Written a week later.")).unwrap();
    w.add_link(Link::new(("event", "report0_v1"), "describes", ("document", "late"))).unwrap();
    let err = w.commit(true).unwrap_err();
    assert_eq!(err.code(), Some("E809"), "{err}");
    // Added again as it was, it is what it was: the address comes back.
    let mut w = amend(&path, None).unwrap();
    w.drop_clinical().unwrap();
    w.add_records(worked_records()).unwrap();
    w.commit(true).unwrap();
    assert_eq!(content_id(&path), before);
}

/// An amendment that leaves the clinical group behind still holds every
/// version the source held: one added again must be what it was (§7.3).
/// Left behind by `amend_into`, the group filled nothing to hold, so a
/// version re-added with a payload it never had was written --- in place,
/// and to another path, whose commit read the output for what was held.
#[test]
fn s7_3_amend_into_without_clinical_holds_a_readded_version() {
    for elsewhere in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = worked(dir.path());
        let out = if elsewhere { dir.path().join("out.medh5") } else { path.clone() };
        let before = std::fs::read(&path).unwrap();
        let mut w = medh5::sample::writer::amend_into(&path, &out, None, &["clinical"]).unwrap();
        w.add_records(worked_records()).unwrap();
        w.add_document(medh5::clinical::model::Document::text("late", "Written a week later.")).unwrap();
        w.add_link(Link::new(("event", "report0_v1"), "describes", ("document", "late"))).unwrap();
        let err = w.commit(true).unwrap_err();
        assert_eq!(err.code(), Some("E809"), "elsewhere {elsewhere}: {err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        if elsewhere {
            assert!(!out.exists());
        }
        // Added again as it was, it is what it was.
        let mut w = medh5::sample::writer::amend_into(&path, &out, None, &["clinical"]).unwrap();
        w.add_records(worked_records()).unwrap();
        w.commit(true).unwrap();
        assert_eq!(content_id(&out), content_id(&path));
    }
}

/// What 1.0 §13.2 covers of an attested payload is held with its bytes: a
/// slope rescales every voxel read, and an edit to it passed as long as the
/// pixels were unchanged.
#[test]
fn s7_3_an_attested_payload_s_attributes_are_held() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let before = std::fs::read(&path).unwrap();
    let mut w = amend(&path, None).unwrap();
    let image = w.root().unwrap().dataset("images/CT_tp0").unwrap();
    image.new_attr::<f64>().create("rescale_slope").unwrap().write_scalar(&2.0).unwrap();
    drop(image);
    let err = w.commit(true).unwrap_err();
    assert_eq!(err.code(), Some("E809"), "{err}");
    assert!(err.to_string().contains("image 'CT_tp0', which event 'ct0' attests"), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn s3_the_clock_is_fixed_once_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let mut w = amend(&path, None).unwrap();
    let err = w.set_clock(Clock::relative("another", "elsewhere")).unwrap_err();
    assert_eq!(err.code(), Some("E802"));
    // 1.1 §3 defines no conversion between clocks; the refusal pointed at a
    // manifest that records "verified conversions", which none does.
    assert!(err.message().contains("no conversion between clocks is defined"), "{}", err.message());
    assert!(!err.message().contains("verified conversion"), "{}", err.message());
    w.abort();
}

#[test]
fn s7_the_commit_refuses_what_the_validator_would() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.medh5");
    let ct = medh5::array::NdArray::from_vec(&[8, 16, 16], vec![0i16; 8 * 16 * 16]).unwrap();
    let mut w = create(&path, Some("bad"), Some("S"), "balanced", &[]).unwrap();
    w.add_grid("ct", &[8, 16, 16], &[2.0, 0.8, 0.8], GridOptions::default()).unwrap();
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default()).unwrap();
    w.set_clock(Clock::relative("c", "baseline")).unwrap();
    w.add_link(Link::new(("event", "ghost"), "describes", ("image", "CT"))).unwrap();
    let err = w.commit(true).unwrap_err();
    assert_eq!(err.code(), Some("E009"), "{err}"); // no event at all
    assert!(!path.exists(), "nothing was written");
}

#[test]
fn s10_augmentation_keeps_payload_digests_and_reports_unknowns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("img.medh5");
    imaging_only(&path);
    let (before, payloads) = (content_id(&path), payload_digests(&path));
    let (events, links, notes) = imaging_events_from_timepoints(&path).unwrap();
    assert!(notes.iter().any(|n| n.contains("days_from_baseline")), "{notes:?}");
    assert!(events.is_empty(), "the default timepoint declares no interval");
    let mut records = ClinicalRecords::new(Descriptor::new(Clock::relative("subject-clock", "enrolment")));
    let mut ct = Event {
        event_id: "ct".into(),
        record_id: "ct".into(),
        kind: "imaging".into(),
        temporal_type: "point".into(),
        effective_start: Some(Bounds::new(0, 24 * HOUR - 1)),
        status: "final".into(),
        ..Default::default()
    };
    ct.timepoint_id = Some("tp0".into());
    records.events.push(ct);
    records.links.push(Link::new(("event", "ct"), "describes", ("image", "CT")));
    records.links.extend(links);
    let out = dir.path().join("augmented.medh5");
    let report = augment(&path, records, Some(&out)).unwrap();
    assert_eq!((report.version_before.as_str(), report.version_after.as_str()), ("1.0", "1.1"));
    assert_ne!(report.content_id_after.as_deref(), Some(before.as_str()));
    assert_eq!(report.unchanged_digests, payloads.len());
    assert!(report.assumptions.iter().any(|a| a.contains("unknown availability")), "{:?}", report.assumptions);
    assert!(report.assumptions.iter().any(|a| a.contains("to a day")), "{:?}", report.assumptions);
    assert_eq!(content_id(&path), before, "the source is untouched");
    assert_eq!(payload_digests(&out), payloads);
}

#[test]
fn s10_a_foreign_clinical_group_is_never_reinterpreted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("img.medh5");
    imaging_only(&path);
    let file = hdf5::File::open_rw(&path).unwrap();
    file.create_group("clinical").unwrap().new_dataset::<u8>().shape([3]).create("site_notes").unwrap();
    drop(file);
    // A 1.0 extension of that name: valid, copied through by amend...
    let before = content_id(&path);
    let mut w = amend(&path, None).unwrap();
    w.commit(true).unwrap();
    assert!(open_sample(&path).unwrap().root.group("clinical").is_ok());
    let _ = before;
    // ...and never taken for the profile.
    let records = ClinicalRecords::new(Descriptor::new(Clock::relative("c", "baseline")));
    let err = augment(&path, records, None).unwrap_err();
    assert_eq!(err.code(), Some("E803"), "{err}");
}

#[test]
fn s10_stripping_is_a_reported_loss() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let out = dir.path().join("imaging.medh5");
    let report = strip(&path, &out).unwrap();
    assert_eq!(report.version_after, "1.0");
    assert_eq!(report.events_removed, 8);
    assert_ne!(report.content_id_after, report.content_id_before);
    let s = open_sample(&out).unwrap();
    assert!(s.clinical().unwrap().is_none());
    assert!(validate_file(&out, "integrity", None).unwrap().ok());
    assert!(strip(&path, &path).is_err(), "never in place");
}

/// A minor this engine does not implement: the one after the newest it does.
fn later_minor() -> String {
    medh5::version::next_minor(medh5::version::LATEST_VERSION).unwrap()
}

fn bump_minor(path: &Path) {
    let file = hdf5::File::open_rw(path).unwrap();
    medh5::h5::attrs::write(&file, "medh5_version", &medh5::h5::AttrValue::Str(later_minor())).unwrap();
}

#[test]
fn s2_a_higher_minor_is_read_as_a_projection_and_never_amended() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    bump_minor(&path);
    let s = open_sample(&path).unwrap();
    assert_eq!(s.support().unwrap(), medh5::version::Support::Projection);
    assert!(s.clinical().unwrap().unwrap().projection);
    drop(s);
    let bytes = std::fs::read(&path).unwrap();
    assert!(matches!(amend(&path, None), Err(Error::Version(_))));
    assert!(matches!(recompress(&path, "archive", None, false), Err(Error::Version(_))));
    let shard = dir.path().join("shard.medh5c");
    assert!(matches!(pack(&[path.as_path()], &shard, None), Err(Error::Version(_))));
    assert_eq!(std::fs::read(&path).unwrap(), bytes, "refused before anything was written");
    let report = validate_file(&path, "semantic", None).unwrap();
    assert!(report.codes().contains(&"W913".to_string()));
    assert!(!validate_file(&path, "strict", None).unwrap().ok(), "strict never calls a projection conforming");
}

#[test]
fn s8_mixed_collections_keep_each_members_version_and_address() {
    let dir = tempfile::tempdir().unwrap();
    let clinical = worked(dir.path());
    let plain = dir.path().join("plain.medh5");
    imaging_only(&plain);
    let shard = dir.path().join("mixed.medh5c");
    pack(&[plain.as_path(), clinical.as_path()], &shard, None).unwrap();
    let c = open_collection(&shard).unwrap();
    assert_eq!(c.version().unwrap(), "1.1");
    assert_eq!(c.get("plain").unwrap().version().unwrap(), "1.0");
    assert_eq!(c.get("worked").unwrap().version().unwrap(), "1.1");
    assert_eq!(c.get("worked").unwrap().content_id().unwrap().unwrap(), content_id(&clinical));
    drop(c);
    let out = dir.path().join("back.medh5");
    extract(&shard, "plain", &out).unwrap();
    assert_eq!(open_sample(&out).unwrap().version().unwrap(), "1.0", "never promoted");
    assert_eq!(content_id(&out), content_id(&plain));
    let out = dir.path().join("back-clinical.medh5");
    extract(&shard, "worked", &out).unwrap();
    assert_eq!(content_id(&out), content_id(&clinical));
    assert!(validate_file(&out, "integrity", None).unwrap().ok());
}

#[test]
fn s8_recompression_keeps_the_content_id() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let before = content_id(&path);
    let result = recompress(&path, "archive", None, false).unwrap();
    assert!(result.ok(), "{:?}", result.to_json());
    assert_eq!(content_id(&path), before);
    assert!(validate_file(&path, "integrity", None).unwrap().ok());
}

#[test]
fn s9_3_at_hour_24_the_preliminary_report_is_the_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    let s = open_sample(&path).unwrap();
    let c = s.clinical().unwrap().unwrap();
    let links: Vec<(usize, &Link)> = c.links.iter().map(|l| (0, l)).collect();
    let at = select(&c.events, &links, 24 * HOUR, &SelectionPolicy::strict()).unwrap();
    assert!(at.certified());
    assert_eq!(at.event_ids(&c.events), ["lab0", "ct0", "report0_v1"]);
    assert!(at.admits(0, "image", "CT_tp0"));
    assert!(at.admits(0, "document", "report0_text_v1"));
    for (kind, id) in [
        ("document", "report0_text_v2"),
        ("image", "CT_tp1"),
        ("annotation", "lesions_tp0"), // drawn after the follow-up
        ("instance", "1"),             // grounded in the report only later
    ] {
        assert!(!at.admits(0, kind, id), "{kind} {id} leaked into the hour-24 input");
    }
    // At hour 48 the revision replaces it; the follow-up is still out.
    let later = select(&c.events, &links, 48 * HOUR, &SelectionPolicy::strict()).unwrap();
    assert_eq!(later.event_ids(&c.events), ["lab0", "ct0", "report0_v2"]);
    // A context window of a day drops the pre-baseline lab.
    let mut day = SelectionPolicy::strict();
    day.context_us = Some(24 * HOUR);
    let windowed = select(&c.events, &links, 24 * HOUR, &day).unwrap();
    assert_eq!(windowed.event_ids(&c.events), ["ct0", "report0_v1"]);
}

#[test]
fn s6_links_from_and_to_are_the_links_leaving_and_arriving() {
    let dir = tempfile::tempdir().unwrap();
    let s = open_sample(&worked(dir.path())).unwrap();
    let clinical = s.clinical().unwrap().unwrap();
    assert!(!clinical.links.is_empty());
    for link in &clinical.links {
        let leaving: Vec<&Link> = clinical.links_from(&link.source_type, &link.source_id).collect();
        assert!(leaving.contains(&link));
        assert!(leaving.iter().all(|l| l.source_type == link.source_type && l.source_id == link.source_id));
        let arriving: Vec<&Link> = clinical.links_to(&link.target_type, &link.target_id).collect();
        assert!(arriving.contains(&link));
        assert!(arriving.iter().all(|l| l.target_type == link.target_type && l.target_id == link.target_id));
    }
    assert_eq!(clinical.links_from("event", "no-such-event").count(), 0);
    assert_eq!(clinical.links_to("image", "no-such-image").count(), 0);
}

/// A link's span is any `uint64` its file states: the validator found where
/// it falls in the text buffer as `start + span`, which wrapped past the
/// buffer's end into it.  A span past the document is E814.
#[test]
fn s6_a_span_past_the_integers_is_e814_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = worked(dir.path());
    {
        let file = medh5::hdf5::File::open_rw(&path).unwrap();
        let links = file.group("clinical/links").unwrap();
        // The second document, which starts past the buffer's first byte:
        // the first starts at 0, and nothing is added to it.
        let ids = links.dataset("source_id/data").unwrap();
        let mut bytes: Vec<u8> = ids.read_raw().unwrap();
        let at = bytes.windows(15).position(|w| w == b"report0_text_v1").unwrap();
        bytes[at + 14] = b'2';
        ids.write(&bytes).unwrap();
        for name in ["source_start", "source_end"] {
            let valid: Vec<u8> = links.dataset(&format!("valid/{name}")).unwrap().read_raw().unwrap();
            let far: Vec<u64> = valid.iter().map(|v| if *v == 1 { u64::MAX } else { 0 }).collect();
            links.dataset(name).unwrap().write(&far).unwrap();
        }
    }
    let report = validate_file(&path, "semantic", None).unwrap();
    assert!(report.codes().contains(&"E814".to_string()), "{:?}", report.codes());
}
