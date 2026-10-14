# Contributing

## Issues & pull requests

We use GitHub issues to track public bugs and feature requests. Please
include the comms_viz command you ran, the PyTorch version that wrote the
traces, and, if you can share them, a trace directory that reproduces the
problem.

We actively welcome pull requests for bug fixes and features.
Please ensure your PR passes tests, adds new tests if appropriate, and clearly
documents and defines any new features and metrics.

## Development

```sh
cargo build --release --locked
cargo test --release --locked --workspace
cargo clippy --release --locked --all-targets
cargo fmt --all
```

Integration tests compare against [insta](https://insta.rs) snapshots under
`crates/comms-viz-cli/tests/integration/snapshots/`. After an intended output
change, rerun the tests with `INSTA_UPDATE=always` and review the snapshot
diff.

## Security

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only
need to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to comms_viz, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
