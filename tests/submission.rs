
#[test]
fn test_reject_submission_local_tests() {
    let scenario_dir = tempfile::tempdir().unwrap();
    let submission_dir = tempfile::tempdir().unwrap();
    
    std::fs::write(scenario_dir.path().join("cape-tests.json"), "[]").unwrap();
    std::fs::write(submission_dir.path().join("cape-tests.json"), "[]").unwrap();

    let result = uplc_cape::submission::verify_submission(
        scenario_dir.path(),
        submission_dir.path(),
        &[],
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Submission-local cape-tests.json is not allowed"));
}
