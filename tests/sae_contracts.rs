use sporse::sae::{standardize, train_ccsa, CcsaTrainConfig, CompositeCode, CompositeCodeConfig};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn config() -> CcsaTrainConfig {
    CcsaTrainConfig {
        code: CompositeCodeConfig::new(2, 2),
        input_dim: 2,
        epochs: 0,
        lr: 0.0,
        temperature: 1.0,
        uniformity_weight: 0.0,
        gumbel_noise: false,
        seed: 7,
    }
}
fn panics_with(f: impl FnOnce(), message: &str) {
    let error = catch_unwind(AssertUnwindSafe(f)).expect_err("invalid input must panic");
    let actual = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(
        actual.contains(message),
        "expected {message:?}, got {actual:?}"
    );
}
#[test]
fn training_rejects_invalid_temperature_even_without_epochs() {
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        let cfg = CcsaTrainConfig {
            temperature,
            ..config()
        };
        panics_with(
            || {
                train_ccsa(&[], &cfg);
            },
            "temperature must be finite and positive with a finite reciprocal",
        );
    }
}
#[test]
fn geometry_is_checked_before_allocation() {
    let code = CompositeCodeConfig::new(usize::MAX, 2);
    panics_with(
        || {
            code.dim();
        },
        "composite code dimensionality overflow",
    );
    assert!(CompositeCode::from_logits(&[], code).is_none());
    let cfg = CcsaTrainConfig { code, ..config() };
    panics_with(
        || {
            train_ccsa(&[], &cfg);
        },
        "composite code dimensionality overflow",
    );
    let cfg = CcsaTrainConfig {
        input_dim: usize::MAX,
        ..config()
    };
    panics_with(
        || {
            train_ccsa(&[], &cfg);
        },
        "encoder matrix size overflow",
    );
    if let Some(too_many) = (u32::MAX as usize).checked_add(2) {
        let cfg = CcsaTrainConfig {
            code: CompositeCodeConfig::new(1, too_many),
            ..config()
        };
        panics_with(
            || {
                train_ccsa(&[], &cfg);
            },
            "code dimensions exceed u32 indices",
        );
    }
}
#[test]
fn training_validates_dimensions_rates_and_all_rows() {
    for code in [
        CompositeCodeConfig::new(0, 2),
        CompositeCodeConfig::new(2, 0),
    ] {
        panics_with(
            || {
                train_ccsa(&[], &CcsaTrainConfig { code, ..config() });
            },
            "code geometry must be positive",
        );
    }
    panics_with(
        || {
            train_ccsa(
                &[],
                &CcsaTrainConfig {
                    input_dim: 0,
                    ..config()
                },
            );
        },
        "input_dim must be positive",
    );
    for value in [-1.0, f32::NAN, f32::INFINITY] {
        panics_with(
            || {
                train_ccsa(
                    &[],
                    &CcsaTrainConfig {
                        lr: value,
                        ..config()
                    },
                );
            },
            "lr must be finite and nonnegative",
        );
        panics_with(
            || {
                train_ccsa(
                    &[],
                    &CcsaTrainConfig {
                        uniformity_weight: value,
                        ..config()
                    },
                );
            },
            "uniformity_weight must be finite and nonnegative",
        );
    }
    for row in [vec![1.0], vec![1.0, 2.0, 3.0]] {
        panics_with(
            || {
                train_ccsa(&[vec![1.0, 2.0], row], &config());
            },
            "training row length must equal input_dim",
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        panics_with(
            || {
                train_ccsa(&[vec![1.0, 2.0], vec![value, 0.0]], &config());
            },
            "training rows must be finite",
        );
    }
}
#[test]
fn logits_require_finite_nonempty_checked_geometry() {
    for code in [
        CompositeCodeConfig::new(0, 2),
        CompositeCodeConfig::new(2, 0),
    ] {
        assert!(CompositeCode::from_logits(&[], code).is_none());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            CompositeCode::from_logits(&[0.0, value], CompositeCodeConfig::new(1, 2)).is_none()
        );
    }
    let code = CompositeCode::from_logits(&[-2.0, -2.0, -3.0, -1.0], config().code).unwrap();
    assert_eq!(code.active(), &[0, 1]);
}
#[test]
fn standardize_rejects_entire_invalid_dataset_before_mutation() {
    for invalid in [
        vec![1.0],
        vec![1.0, 2.0, 3.0],
        vec![f32::INFINITY, 1.0],
        vec![f32::NAN, 1.0],
    ] {
        let message = if invalid.len() == 2 {
            "standardize rows must be finite"
        } else {
            "standardize row length must equal input_dim"
        };
        let mut data = vec![vec![-2.0, 3.0], invalid];
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let before = bits(&data);
        panics_with(
            || {
                standardize(&mut data, 2);
            },
            message,
        );
        assert_eq!(bits(&data), before);
    }
}
#[test]
fn valid_empty_zero_epoch_and_negative_inputs_remain_supported() {
    let data = vec![vec![-1.0, 2.0], vec![3.0, -4.0]];
    let initial = train_ccsa(&data, &config());
    let empty = train_ccsa(
        &[],
        &CcsaTrainConfig {
            epochs: 2,
            ..config()
        },
    );
    assert_eq!(initial.reconstruct(&data[0]), empty.reconstruct(&data[0]));
    assert!(initial.reconstruction_mse(&data).is_finite());
    assert_eq!(empty.reconstruction_mse(&[]), 0.0);
    assert!(initial.encode(&[1.0]).is_none());
    assert!(catch_unwind(|| initial.reconstruct(&[1.0])).is_err());
    assert!(catch_unwind(|| initial.reconstruction_mse(&[vec![1.0]])).is_err());
    assert_eq!(standardize(&mut [vec![]], 0), (vec![], vec![]));
    assert_eq!(standardize(&mut [], 2), (vec![0.0; 2], vec![1e-6; 2]));
    let mut standardized = data;
    let (mean, std) = standardize(&mut standardized, 2);
    assert_eq!(mean, vec![1.0, -1.0]);
    assert_eq!(std, vec![2.0, 3.0]);
    assert_eq!(standardized, vec![vec![-1.0, 1.0], vec![1.0, -1.0]]);
}
