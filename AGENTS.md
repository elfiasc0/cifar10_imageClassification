# Repository Guidelines

## Project Structure & Assets

This repository currently contains project reference material rather than application source code. The two PDFs in the root hold the comparative architecture analysis and project information. The `pics/` directory contains image assets in JPEG, PNG, and HEIC formats. Keep these materials in their existing locations unless a project change calls for a clear reorganization; use descriptive names for any new documents or assets.

## Build, Test, and Development

There is no build system, package manifest, or automated test suite in the repository at present. No build or test command is defined. For content-only changes, review the affected document or image directly and confirm filenames and links are correct. If code or scripts are introduced, document the setup and verification commands alongside them.

## Style & Naming

Write documentation in concise Markdown with descriptive headings and relative links to repository files. Preserve original source filenames for supplied materials. Name new assets with a brief, meaningful subject and an appropriate extension (for example, `architecture-diagram.png`); avoid ambiguous names such as `new-file` or `final2`.

## Testing Guidelines

No test framework or coverage target is configured. Validate document edits by checking Markdown structure and referenced paths. For image additions, ensure the file opens and the chosen format is supported by the intended consumers. Do not convert or recompress existing assets without a project need.

## Commits & Pull Requests

Git history is not available in this checkout, so no repository-specific commit convention can be inferred. Use short imperative commit subjects (for example, `Add architecture reference diagram`). Pull requests should summarize the files changed and their purpose, identify any relevant project task, and include a preview or screenshot when a visual asset or rendered document is materially changed.

## Sensitive Information

Before adding material, check that it is intended for this repository and contains no private or identifying information that should not be shared. Keep original reference materials intact when possible.
