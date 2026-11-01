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
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --ref 59e49bf2f1c0f83e18ec3c613a3e94ca36f366ef
```

- [Setup and configuration](https://github.com/electricsheephq/lcm-x/blob/59e49bf2f1c0f83e18ec3c613a3e94ca36f366ef/README.md)
- [Embedding setup](https://github.com/electricsheephq/lcm-x/blob/59e49bf2f1c0f83e18ec3c613a3e94ca36f366ef/docs/embeddings-setup.md)
- [Catalog payload and provenance](https://github.com/electricsheephq/lcm-x/blob/59e49bf2f1c0f83e18ec3c613a3e94ca36f366ef/docs/operations/catalog-payload.md)
- [License](https://github.com/electricsheephq/lcm-x/blob/59e49bf2f1c0f83e18ec3c613a3e94ca36f366ef/LICENSE)
