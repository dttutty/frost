# Model–Runtime Boundary

FROST models declare requirements; the runtime decides how to satisfy them.
`model=<name>` selects one `ModelRecipe`, which combines component configuration
with a `ModelContract`. The launch layer compiles that declaration into an
immutable `ExecutionPlan` before constructing host or worker runtime objects.

## Model ownership

Code under `src/frost/models/` owns differentiable computation only:

- encoders return `EncoderResult`;
- `RepresentationAdapter` converts node-aligned or pair-aligned embeddings to
  `EdgeRepresentations`;
- `AdditiveLinkHead` and `ConcatLinkHead` score explicit edge endpoints;
- `TemporalLinkModel` composes memory update, graph encoding, representation,
  and scoring.

Models do not create queues, CUDA events, P2P tasks, shared-memory writes, or
branch on concrete model class names.

## Runtime ownership

The compiled plan drives runtime policy:

- samplers and RingBuf use `RootExpansion`;
- `SampledLayoutView.select(NodeSet)` resolves graph-layout scopes;
- `AccessPlanAnalyzer` derives feature, memory, and mailbox planner inputs;
- `DynamicStateCoordinator` captures stale state and publishes updater output;
- `NeighborhoodMailboxRouter` implements neighborhood mailbox delivery.

The coordinator attaches a forward hook to the memory updater. Publication can
therefore start as soon as updated state exists and overlap encoder computation,
without putting runtime dependencies in the model.

## Extension rule

A new model must first express its needs with the existing contract vocabulary.
Add a new semantic contract/plan type only when a requirement cannot be expressed
by the current node sets, root expansion, state contract, representation kind,
or scoring kind. Do not add target-string checks or model-named runtime branches.
