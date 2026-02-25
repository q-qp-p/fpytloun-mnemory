.PHONY: ui-build ui-watch ui-vendor ui-all

# Tailwind CSS standalone CLI binary.
# Download from: https://github.com/tailwindlabs/tailwindcss/releases/latest
# macOS ARM:  tailwindcss-macos-arm64
# macOS x64:  tailwindcss-macos-x64
# Linux x64:  tailwindcss-linux-x64
# Linux ARM:  tailwindcss-linux-arm64
TAILWIND_BIN ?= tailwindcss

# Build minified Tailwind CSS from input.css
# Run this before committing if you changed any Tailwind classes in HTML/JS.
ui-build:
	$(TAILWIND_BIN) -c mnemory/ui/tailwind.config.js \
		-i mnemory/ui/src/input.css \
		-o mnemory/ui/static/css/app.css --minify

# Watch mode — rebuilds CSS on file changes during development
ui-watch:
	$(TAILWIND_BIN) -c mnemory/ui/tailwind.config.js \
		-i mnemory/ui/src/input.css \
		-o mnemory/ui/static/css/app.css --watch

# Download vendored JS libraries (run once, committed to repo)
ui-vendor:
	mkdir -p mnemory/ui/static/vendor
	curl -sL -o mnemory/ui/static/vendor/alpine.min.js \
		"https://unpkg.com/alpinejs@3.14.9/dist/cdn.min.js"
	curl -sL -o mnemory/ui/static/vendor/chart.umd.min.js \
		"https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"
	curl -sL -o mnemory/ui/static/vendor/d3.min.js \
		"https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
	@echo "Vendored JS libraries downloaded."

# Full setup: download vendors + build CSS
ui-all: ui-vendor ui-build
	@echo "UI assets ready."
