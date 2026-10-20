//! The vocabulary registry shared by threads, in a process of its own: a
//! name registered here would show in the unit tests' `available()`.

use std::sync::atomic::{AtomicBool, Ordering};

use medh5::labels::registry::{describe, load, register, unregister};

#[test]
fn s5_1_describe_takes_the_registered_vocabularies_in_one_step() {
    // A name unregistered while `describe` runs is described or absent;
    // listed first and looked up after, it was sometimes E305.
    let ls = load("binary-foreground").unwrap();
    let stop = AtomicBool::new(false);
    // Stops the other thread however this one ends: a failed `describe`
    // would otherwise leave the scope waiting for it.
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    std::thread::scope(|s| {
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                register("churn", ls.clone());
                unregister("churn");
            }
        });
        let _stop = Stop(&stop);
        for _ in 0..2_000 {
            assert!(describe().unwrap().contains_key("binary-foreground"));
        }
    });
}

#[test]
fn s5_1_describe_reads_a_registered_name_as_load_does() {
    // A vocabulary registered under a bundled name is the one `load` gives.
    let mut own = load("brats-subregions").unwrap();
    own.version = "9.9.9".into();
    register("brats-subregions", own);
    let described = describe().unwrap();
    unregister("brats-subregions");
    assert_eq!(described["brats-subregions"]["version"], "9.9.9");
    assert_eq!(describe().unwrap()["brats-subregions"]["version"], load("brats-subregions").unwrap().version);
}
