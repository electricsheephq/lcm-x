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
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --ref f77d6ed98c8f2e896851b31720876e4c9d10f7b4
```

- [Setup and configuration](https://github.com/electricsheephq/lcm-x/blob/f77d6ed98c8f2e896851b31720876e4c9d10f7b4/README.md)
- [Embedding setup](https://github.com/electricsheephq/lcm-x/blob/f77d6ed98c8f2e896851b31720876e4c9d10f7b4/docs/embeddings-setup.md)
- [Catalog payload and provenance](https://github.com/electricsheephq/lcm-x/blob/f77d6ed98c8f2e896851b31720876e4c9d10f7b4/docs/operations/catalog-payload.md)
- [License](https://github.com/electricsheephq/lcm-x/blob/f77d6ed98c8f2e896851b31720876e4c9d10f7b4/LICENSE)
