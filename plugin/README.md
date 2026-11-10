# LCM-X for Hermes

LCM-X keeps the live context bounded, preserves raw conversation history, and
provides cited retrieval across sessions. This is the runtime catalog payload.

```sh
hermes plugins install hermes-lcm-x
hermes plugins enable hermes-lcm-x
```

Set `context.engine: lcm-x` in your Hermes configuration, then restart Hermes.

Local semantic recall needs the companion below; without it, recall uses full text.

```sh
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --ref aa1f63cdb4c520e1381fd05bc3c553d3e2f87dce
```

- [Setup and configuration](https://github.com/electricsheephq/lcm-x/blob/aa1f63cdb4c520e1381fd05bc3c553d3e2f87dce/README.md)
- [Embedding setup](https://github.com/electricsheephq/lcm-x/blob/aa1f63cdb4c520e1381fd05bc3c553d3e2f87dce/docs/embeddings-setup.md)
- [Catalog payload and provenance](https://github.com/electricsheephq/lcm-x/blob/aa1f63cdb4c520e1381fd05bc3c553d3e2f87dce/docs/operations/catalog-payload.md)
- [License](https://github.com/electricsheephq/lcm-x/blob/aa1f63cdb4c520e1381fd05bc3c553d3e2f87dce/LICENSE)
