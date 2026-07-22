# Incoming text

Files in this directory are staged editorial inputs. Create them with
`make new-content` so every draft records its module, target, content type,
source language, review status, citation status, and rights-review status.

An incoming file must keep these sections:

- `Purpose`
- `Learning objectives`
- `Draft text`
- `Sources to verify`
- `Integration notes`

The validator ignores this README and checks every other Markdown file in this
directory. An incoming draft is not part of either book TOC.
