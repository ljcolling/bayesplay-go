## About

`bayesplay-go` is a Go WASM library for computing Bayes factors for simple
models. It's the Go counterpart of the [bayesplay
R package](https://bayesplay.github.io/bayesplay/) and it powers the
[bayesplay webapp](https://bayesplay.colling.net.nz).

### Building

To build the WASM file simply run:

```bash
make
```

Or to run the tests run:

```bash
make tests
```

A pre-build WASM library is also available in `./dist/main.wasm`

### Components

The two internal modules `pkg/bayesfactor` and `pkg/distributions` provide
functionality for computing Bayes factors and statistical distributions,
respectively. These can be re-used in standalone projects such, for
example, building other package for statistical computations. The main
`cmd/bayesplay` module primary serves as a bridge between go/WASM and
Javascript and is likely to only be useful in the context of the 
[bayesplay webapp](https://bayesplay.colling.net.nz).


