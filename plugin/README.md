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
hermes plugins install electricsheephq/lcm-x/companions/hermes-lcm-x-local-embeddings --ref 3c24d8725e60d003674250492d38646efa18be28
```

- [Setup and configuration](https://github.com/electricsheephq/lcm-x/blob/3c24d8725e60d003674250492d38646efa18be28/README.md)
- [Embedding setup](https://github.com/electricsheephq/lcm-x/blob/3c24d8725e60d003674250492d38646efa18be28/docs/embeddings-setup.md)
- [Catalog payload and provenance](https://github.com/electricsheephq/lcm-x/blob/3c24d8725e60d003674250492d38646efa18be28/docs/operations/catalog-payload.md)
- [License](https://github.com/electricsheephq/lcm-x/blob/3c24d8725e60d003674250492d38646efa18be28/LICENSE)
