.PHONY: wasm api check brand site preview test test-wasm all

wasm:
	wasm-pack build core --target web --release --out-dir ../app/pkg

# Rust API documentation, copied into the site as api/ (the {{< src >}} shortcodes link into it)
api:
	cargo doc --no-deps --release --manifest-path core/Cargo.toml
	rm -rf api && cp -r core/target/doc api

# every provenance link of the text resolves (needs api/)
check: api
	python3 scripts/check_links.py

# app/css/brand.css from _brand.yml (also run by quarto as a pre-render step)
brand:
	python3 scripts/brand_to_css.py

site:
	quarto render

preview:
	quarto preview

test:
	cargo test --release --manifest-path core/Cargo.toml
	node scripts/test_lab.mjs
	node scripts/test_two_point.mjs

test-wasm: wasm
	node scripts/test_wasm.mjs
	node scripts/test_worker.mjs

all: test wasm test-wasm check site
