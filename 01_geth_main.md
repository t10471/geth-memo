
# makeFullNode

## makeFullNode > makeConfigNode
configを作成

## config
```
	cfg := gethConfig{
		Eth:       eth.DefaultConfig,
		Shh:       whisper.DefaultConfig,
		Node:      defaultNodeConfig(),
		Dashboard: dashboard.DefaultConfig,
	}
```

## makeFullNode > makeConfigNode > SetNodeConfig
```
	SetP2PConfig(ctx, &cfg.P2P)
	setIPC(ctx, cfg)
	setHTTP(ctx, cfg)
	setWS(ctx, cfg)
	setNodeUserIdent(ctx, cfg)
```

## makeFullNode > makeConfigNode > SetNodeConfig > SetP2PConfig
```
	setNodeKey(ctx, cfg)
	setNAT(ctx, cfg)
	setListenAddress(ctx, cfg)
	// 起動nodeの決定?
	setBootstrapNodes(ctx, cfg)
	setBootstrapNodesV5(ctx, cfg)
	// light mode (client or server) の判定
	// DiscoveryV5を使うか判定
	// developer mode (p2pしない) 判定
```

### makeFullNode > makeConfigNode > SetNodeConfig > SetP2PConfig > setIPC
プライベートネットの場合に使う?
IPCPathを設定する

## makeFullNode > makeConfigNode > node.New
### AccountManagerを作る
makeAccountManager(conf)

BackendとKeystoreを作成
下のデータを作成している
```go
	// 処理
	backends := []accounts.Backend{
		keystore.NewKeyStore(keydir, scryptN, scryptP),
	}

// BackendとKeyStoreの型
type Backend interface {
	Wallets() []Wallet
	Subscribe(sink chan<- WalletEvent) event.Subscription
}
type KeyStore struct {
	storage  keyStore                     // Storage backend, might be cleartext or encrypted
	cache    *accountCache                // In-memory account cache over the filesystem storage
	changes  chan struct{}                // Channel receiving change notifications from the cache
	unlocked map[common.Address]*unlocked // Currently unlocked account (decrypted private keys)

	wallets     []accounts.Wallet       // Wallet wrappers around the individual key files ImpleはKeystoreWallet
	updateFeed  event.Feed              // Event feed to notify wallet additions/removals
	updateScope event.SubscriptionScope // Subscription scope tracking current live listeners
	updating    bool                    // Whether the event notification loop is running

	mu sync.RWMutex
}
type accountCache struct {
	keydir   string
	watcher  *watcher
	mu       sync.Mutex
	all      accountsByURL
	byAddr   map[common.Address][]accounts.Account
	throttle *time.Timer
	notify   chan struct{}
	fileC    fileCache
}

type Wallet interface {
	URL() URL
	Status() (string, error)
	Open(passphrase string) error
	Close() error
	Accounts() []Account
	Contains(account Account) bool
	Derive(path DerivationPath, pin bool) (Account, error)
	SelfDerive(base DerivationPath, chain ethereum.ChainStateReader)
	SignHash(account Account, hash []byte) ([]byte, error)
	SignTx(account Account, tx *types.Transaction, chainID *big.Int) (*types.Transaction, error)
	SignHashWithPassphrase(account Account, passphrase string, hash []byte) ([]byte, error)
	SignTxWithPassphrase(account Account, passphrase string, tx *types.Transaction, chainID *big.Int) (*types.Transaction, error)
}
// 上のImplがkeystoreWallet
type keystoreWallet struct {
	account  accounts.Account // Single account contained in this wallet
	keystore *KeyStore        // Keystore where the account originates from
}
type Account struct {
	Address common.Address `json:"address"` // Ethereum account address derived from the key
	URL     URL            `json:"url"`     // Optional resource locator within a backend
}
```

Backend KeyStoreを使ってAccountManagerを作成する  
AccountManagerはWalletのイベント監視をしている

```go

	// Subscribe to wallet notifications from all backends
	updates := make(chan WalletEvent, 4*len(backends))

	subs := make([]event.Subscription, len(backends))
	for i, backend := range backends {
		subs[i] = backend.Subscribe(updates)
	}
	// Assemble the account manager and return
	am := &Manager{
		backends: make(map[reflect.Type][]Backend),
		updaters: subs,
		updates:  updates,
		wallets:  wallets,
		quit:     make(chan chan error),
	}

	go am.update()

type Manager struct {
	backends map[reflect.Type][]Backend // Index of backends currently registered
	updaters []event.Subscription       // Wallet update subscriptions for all backends
	updates  chan WalletEvent           // Subscription sink for backend wallet changes
	wallets  []Wallet                   // Cache of all wallets from all registered backends

	feed event.Feed // Wallet feed notifying of arrivals/departures

	quit chan chan error
	lock sync.RWMutex
}

func (am *Manager) update() {
	// Close all subscriptions when the manager terminates
	defer func() {
		am.lock.Lock()
		for _, sub := range am.updaters {
			sub.Unsubscribe()
		}
		am.updaters = nil
		am.lock.Unlock()
	}()

	// Loop until termination
	for {
		select {
		case event := <-am.updates:
			// Wallet event arrived, update local cache
			am.lock.Lock()
			switch event.Kind {
			case WalletArrived:
				am.wallets = merge(am.wallets, event.Wallet)
			case WalletDropped:
				am.wallets = drop(am.wallets, event.Wallet)
			}
			am.lock.Unlock()

			// Notify any listeners of the event
			am.feed.Send(event)

		case errc := <-am.quit:
			// Manager terminating, return
			errc <- nil
			return
		}
	}
}

```

BackendのSubscribeはKeyStoreの。  
WalletEventをFeedにしてSubscribeしてSubscriptionにして返す　　
定期的にWalletを更新する
```go
func (ks *KeyStore) Subscribe(sink chan<- accounts.WalletEvent) event.Subscription {
	ks.mu.Lock()
	defer ks.mu.Unlock()
	sub := ks.updateScope.Track(ks.updateFeed.Subscribe(sink))
	if !ks.updating {
		ks.updating = true
		go ks.updater()
	}
	return sub
}

// Feed implements one-to-many subscriptions where the carrier of events is a channel.
// Values sent to a Feed are delivered to all subscribed channels simultaneously.
//
// Feeds can only be used with a single type. The type is determined by the first Send or
// Subscribe operation. Subsequent calls to these methods panic if the type does not
// match.
//
// The zero value is ready to use.
type Feed struct {
	once      sync.Once        // ensures that init only runs once
	sendLock  chan struct{}    // sendLock has a one-element buffer and is empty when held.It protects sendCases.
	removeSub chan interface{} // interrupts Send
	sendCases caseList         // the active set of select cases used by Send

	// The inbox holds newly subscribed channels until they are added to sendCases.
	mu     sync.Mutex
	inbox  caseList
	etype  reflect.Type
	closed bool
}


func (f *Feed) Subscribe(channel interface{}) Subscription {
	f.once.Do(f.init)

	chanval := reflect.ValueOf(channel)
	chantyp := chanval.Type()
	if chantyp.Kind() != reflect.Chan || chantyp.ChanDir()&reflect.SendDir == 0 {
		panic(errBadChannel)
	}
	sub := &feedSub{feed: f, channel: chanval, err: make(chan error, 1)}

	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.typecheck(chantyp.Elem()) {
		panic(feedTypeError{op: "Subscribe", got: chantyp, want: reflect.ChanOf(reflect.SendDir, f.etype)})
	}
	cas := reflect.SelectCase{Dir: reflect.SelectSend, Chan: chanval}
	f.inbox = append(f.inbox, cas)
	return sub
}

func (ks *KeyStore) updater() {
	for {
		select {
		case <-ks.changes:
		case <-time.After(walletRefreshCycle):
		}
		// Walletの状態を調べて、walletsの状態を更新して
		// WalletDropped WalletArrivedのイベントを送信する(ks.updateFeed.Send(event))
		ks.refreshWallets()

		ks.mu.Lock()
		if ks.updateScope.Count() == 0 {
			ks.updating = false
			ks.mu.Unlock()
			return
		}
		ks.mu.Unlock()
	}
}
// refreshWallets retrieves the current account list and based on that does any
// necessary wallet refreshes.
func (ks *KeyStore) refreshWallets() {
	// Retrieve the current list of accounts
	ks.mu.Lock()
	accs := ks.cache.accounts()

	// Transform the current list of wallets into the new one
	wallets := make([]accounts.Wallet, 0, len(accs))
	events := []accounts.WalletEvent{}

	for _, account := range accs {
		// Drop wallets while they were in front of the next account
		for len(ks.wallets) > 0 && ks.wallets[0].URL().Cmp(account.URL) < 0 {
			events = append(events, accounts.WalletEvent{Wallet: ks.wallets[0], Kind: accounts.WalletDropped})
			ks.wallets = ks.wallets[1:]
		}
		// If there are no more wallets or the account is before the next, wrap new wallet
		if len(ks.wallets) == 0 || ks.wallets[0].URL().Cmp(account.URL) > 0 {
			wallet := &keystoreWallet{account: account, keystore: ks}

			events = append(events, accounts.WalletEvent{Wallet: wallet, Kind: accounts.WalletArrived})
			wallets = append(wallets, wallet)
			continue
		}
		// If the account is the same as the first wallet, keep it
		if ks.wallets[0].Accounts()[0] == account {
			wallets = append(wallets, ks.wallets[0])
			ks.wallets = ks.wallets[1:]
			continue
		}
	}
	// Drop any leftover wallets and set the new batch
	for _, wallet := range ks.wallets {
		events = append(events, accounts.WalletEvent{Wallet: wallet, Kind: accounts.WalletDropped})
	}
	ks.wallets = wallets
	ks.mu.Unlock()

	// Fire all wallet events and return
	for _, event := range events {
		ks.updateFeed.Send(event)
	}
}
```
Nodeを返す　　
amはaccountManager　　
ephemeralKeystoreはkeystoreのディレクトリパス
```go
	return &Node{
		accman:            am,
		ephemeralKeystore: ephemeralKeystore,
		config:            conf,
		serviceFuncs:      []ServiceConstructor{},
		ipcEndpoint:       conf.IPCEndpoint(),
		httpEndpoint:      conf.HTTPEndpoint(),
		wsEndpoint:        conf.WSEndpoint(),
		eventmux:          new(event.TypeMux),
		log:               conf.Logger,
	}, nil

```

## makeFullNode > makeConfigNode > SetEthConfig
```go
	ks := stack.AccountManager().Backends(keystore.KeyStoreType)[0].(*keystore.KeyStore)
	// マイニングの報酬を受け取るアドレスを設定する
	setEtherbase(ctx, ks, cfg)
	// GPO は Gas Price Oracle (Gas Priceを決める) gasprice.Configの設定をする
	setGPO(ctx, &cfg.GPO)
	setTxPool(ctx, &cfg.TxPool)
	setEthash(ctx, cfg)

// eth/gasprice/gasprice.go
type Config struct {
		Blocks     int
		Percentile int
		Default    *big.Int `toml:",omitempty"`
}

type TxPoolConfig struct {
	NoLocals  bool          // Whether local transaction handling should be disabled
	Journal   string        // Journal of local transactions to survive node restarts
	Rejournal time.Duration // Time interval to regenerate the local transaction journal

	PriceLimit uint64 // Minimum gas price to enforce for acceptance into the pool
	PriceBump  uint64 // Minimum price bump percentage to replace an already existing transaction (nonce)

	AccountSlots uint64 // Minimum number of executable transaction slots guaranteed per account
	GlobalSlots  uint64 // Maximum number of executable transaction slots for all accounts
	AccountQueue uint64 // Maximum number of non-executable transaction slots permitted per account
	GlobalQueue  uint64 // Maximum number of non-executable transaction slots for all accounts

	Lifetime time.Duration // Maximum amount of time non-executable transaction are queued
}
var DefaultTxPoolConfig = TxPoolConfig{
	Journal:   "transactions.rlp",
	Rejournal: time.Hour,

	PriceLimit: 1,
	PriceBump:  10,

	AccountSlots: 16,
	GlobalSlots:  4096,
	AccountQueue: 64,
	GlobalQueue:  1024,

	Lifetime: 3 * time.Hour,
}

// Config are the configuration parameters of the ethash.
type Config struct {
	CacheDir       string
	CachesInMem    int
	CachesOnDisk   int
	DatasetDir     string
	DatasetsInMem  int
	DatasetsOnDisk int
	PowMode        Mode
}

// Ethash is a consensus engine based on proot-of-work implementing the ethash
// algorithm.
type Ethash struct {
	config Config

	caches   *lru // In memory caches to avoid regenerating too often
	datasets *lru // In memory datasets to avoid regenerating too often

	// Mining related fields
	rand     *rand.Rand    // Properly seeded random source for nonces
	threads  int           // Number of threads to mine on if mining
	update   chan struct{} // Notification channel to update mining parameters
	hashrate metrics.Meter // Meter tracking the average hashrate

	// The fields below are hooks for testing
	shared    *Ethash       // Shared PoW verifier to avoid cache regeneration
	fakeFail  uint64        // Block number which fails PoW check even in fake mode
	fakeDelay time.Duration // Time delay to sleep for before returning from verify

	lock sync.Mutex // Ensures thread safety for the in-memory caches and mining fields
}

```

SyncModeの決定

```go
	switch {
	case ctx.GlobalIsSet(SyncModeFlag.Name):
		cfg.SyncMode = *GlobalTextMarshaler(ctx, SyncModeFlag.Name).(*downloader.SyncMode)
	case ctx.GlobalBool(FastSyncFlag.Name):
		cfg.SyncMode = downloader.FastSync
	case ctx.GlobalBool(LightModeFlag.Name):
		cfg.SyncMode = downloader.LightSync
	}
```
などの設定が続く



# startNode

stack は Node

## startNode > utils.StartNode(stack)

## startNode > utils.StartNode(stack) > stack.Start()

データを保存するディレクトリをセットアップする
openDataDir

p2pサーバを起動する
```go
type Server struct {
	// Config fields may not be modified while the server is running.
	Config

	// Hooks for testing. These are useful because we can inhibit
	// the whole protocol stack.
	newTransport func(net.Conn) transport
	newPeerHook  func(*Peer)

	lock    sync.Mutex // protects running
	running bool

	ntab         discoverTable
	listener     net.Listener
	ourHandshake *protoHandshake
	lastLookup   time.Time
	DiscV5       *discv5.Network

	// These are for Peers, PeerCount (and nothing else).
	peerOp     chan peerOpFunc
	peerOpDone chan struct{}

	quit          chan struct{}
	addstatic     chan *discover.Node
	removestatic  chan *discover.Node
	posthandshake chan *conn
	addpeer       chan *conn
	delpeer       chan peerDrop
	loopWG        sync.WaitGroup // loop, listenLoop
	peerFeed      event.Feed
	log           log.Logger
}

type Config struct {
	// This field must be set to a valid secp256k1 private key.
	PrivateKey *ecdsa.PrivateKey `toml:"-"`

	// MaxPeers is the maximum number of peers that can be
	// connected. It must be greater than zero.
	MaxPeers int

	// MaxPendingPeers is the maximum number of peers that can be pending in the
	// handshake phase, counted separately for inbound and outbound connections.
	// Zero defaults to preset values.
	MaxPendingPeers int `toml:",omitempty"`

	// DialRatio controls the ratio of inbound to dialed connections.
	// Example: a DialRatio of 2 allows 1/2 of connections to be dialed.
	// Setting DialRatio to zero defaults it to 3.
	DialRatio int `toml:",omitempty"`

	// NoDiscovery can be used to disable the peer discovery mechanism.
	// Disabling is useful for protocol debugging (manual topology).
	NoDiscovery bool

	// DiscoveryV5 specifies whether the the new topic-discovery based V5 discovery
	// protocol should be started or not.
	DiscoveryV5 bool `toml:",omitempty"`

	// Name sets the node name of this server.
	// Use common.MakeName to create a name that follows existing conventions.
	Name string `toml:"-"`

	// BootstrapNodes are used to establish connectivity
	// with the rest of the network.
	BootstrapNodes []*discover.Node

	// BootstrapNodesV5 are used to establish connectivity
	// with the rest of the network using the V5 discovery
	// protocol.
	BootstrapNodesV5 []*discv5.Node `toml:",omitempty"`

	// Static nodes are used as pre-configured connections which are always
	// maintained and re-connected on disconnects.
	StaticNodes []*discover.Node

	// Trusted nodes are used as pre-configured connections which are always
	// allowed to connect, even above the peer limit.
	TrustedNodes []*discover.Node

	// Connectivity can be restricted to certain IP networks.
	// If this option is set to a non-nil value, only hosts which match one of the
	// IP networks contained in the list are considered.
	NetRestrict *netutil.Netlist `toml:",omitempty"`

	// NodeDatabase is the path to the database containing the previously seen
	// live nodes in the network.
	NodeDatabase string `toml:",omitempty"`

	// Protocols should contain the protocols supported
	// by the server. Matching protocols are launched for
	// each peer.
	Protocols []Protocol `toml:"-"`

	// If ListenAddr is set to a non-nil address, the server
	// will listen for incoming connections.
	//
	// If the port is zero, the operating system will pick a port. The
	// ListenAddr field will be updated with the actual address when
	// the server is started.
	ListenAddr string

	// If set to a non-nil value, the given NAT port mapper
	// is used to make the listening port available to the
	// Internet.
	NAT nat.Interface `toml:",omitempty"`

	// If Dialer is set to a non-nil value, the given Dialer
	// is used to dial outbound peer connections.
	Dialer NodeDialer `toml:"-"`

	// If NoDial is true, the server will not dial any peers.
	NoDial bool `toml:",omitempty"`

	// If EnableMsgEvents is set then the server will emit PeerEvents
	// whenever a message is sent to or received from a peer
	EnableMsgEvents bool

	// Logger is a custom logger to use with the p2p.Server.
	Logger log.Logger `toml:",omitempty"`
}
```

Resiterで登録したprotocolのサービスをServer.Config.Protcolsに登録する

## startNode > utils.StartNode(stack) > stack.Start() > running.Start()
runningはp2p.Server

### discoverを設定する
discover.Configをセットアップしてntab(node table)に登録する  
udpのセットアップをしてListenする

dialStateを作成する
```go
func newDialState(static []*discover.Node, bootnodes []*discover.Node, ntab discoverTable, maxdyn int, netrestrict *netutil.Netlist) *dialstate {
	s := &dialstate{
		maxDynDials: maxdyn,
		ntab:        ntab,
		netrestrict: netrestrict,
		static:      make(map[discover.NodeID]*dialTask),
		dialing:     make(map[discover.NodeID]connFlag),
		bootnodes:   make([]*discover.Node, len(bootnodes)),
		randomNodes: make([]*discover.Node, maxdyn/2),
		hist:        new(dialHistory),
	}
	copy(s.bootnodes, bootnodes)
	for _, n := range static {
		s.addStatic(n)
	}
	return s
}

```

srv.startListening()
tcpのセットアップしてListenする

go srv.run(dialer)
p2pの接続監視ループ

登録してあるサービスをStartさせる

Ethereum fullNodeの場合

BloomFillterの受信スレッドを起動

## (s *Ethereum) Start(srvr *p2p.Server) > s.protocolManager.Start(maxPeers)
```go
func (pm *ProtocolManager) Start(maxPeers int) {
	pm.maxPeers = maxPeers

	// broadcast transactions
	pm.txCh = make(chan core.TxPreEvent, txChanSize)
	pm.txSub = pm.txpool.SubscribeTxPreEvent(pm.txCh)
	// transaction broadcastを監視
	go pm.txBroadcastLoop()

	// broadcast mined blocks
	pm.minedBlockSub = pm.eventMux.Subscribe(core.NewMinedBlockEvent{})
	// マイニングされたbloadcastを監視
	go pm.minedBroadcastLoop()

	// start sync handlers
	// blockの同期をとる
	go pm.syncer()
	// transactionの同期をとる
	go pm.txsyncLoop()
}

```

## (n *Node) Start() startRPC
n.startInProc(apis)  
unixドメインソケットを開いてサーバを起動  
startIPC(apis)  
tcpソケットを開いてhttpサーバを起動  
startHTTP(n.httpEndpoint, apis, n.config.HTTPModules, n.config.HTTPCors, n.config.HTTPVirtualHosts)  
tcpソケットを開いてwebsocketサーバを起動  
n.startWS(n.wsEndpoint, apis, n.config.WSModules, n.config.WSOrigins, n.config.WSExposeAll)

startInProc
```go
func (n *Node) startInProc(apis []rpc.API) error {
	// Register all the APIs exposed by the services
	handler := rpc.NewServer()
	for _, api := range apis {
		if err := handler.RegisterName(api.Namespace, api.Service); err != nil {
			return err
		}
		n.log.Debug("InProc registered", "service", api.Service, "namespace", api.Namespace)
	}
	n.inprocHandler = handler
	return nil
}

func NewServer() *Server {
	server := &Server{
		services: make(serviceRegistry),
		codecs:   set.New(),
		run:      1,
	}

	// register a default service which will provide meta information about the RPC service such as the services and
	// methods it offers.
	rpcService := &RPCService{server}
	server.RegisterName(MetadataApi, rpcService)

	return server
}
```
startIPC
```go
func (n *Node) startIPC(apis []rpc.API) error {
	if n.ipcEndpoint == "" {
		return nil // IPC disabled.
	}
	listener, handler, err := rpc.StartIPCEndpoint(n.ipcEndpoint, apis)
	if err != nil {
		return err
	}
	n.ipcListener = listener
	n.ipcHandler = handler
	n.log.Info("IPC endpoint opened", "url", n.ipcEndpoint)
	return nil
}

func StartIPCEndpoint(ipcEndpoint string, apis []API) (net.Listener, *Server, error) {
	// Register all the APIs exposed by the services.
	// 上のNewServerと一緒
	handler := NewServer()
	for _, api := range apis {
		if err := handler.RegisterName(api.Namespace, api.Service); err != nil {
			return nil, nil, err
		}
		log.Debug("IPC registered", "namespace", api.Namespace)
	}
	// All APIs registered, start the IPC listener.
	listener, err := ipcListen(ipcEndpoint)
	if err != nil {
		return nil, nil, err
	}
	go handler.ServeListener(listener)
	return listener, handler, nil
}

```
startHTTP
```go
func (n *Node) startHTTP(endpoint string, apis []rpc.API, modules []string, cors []string, vhosts []string) error {
	// Short circuit if the HTTP endpoint isn't being exposed
	if endpoint == "" {
		return nil
	}
	listener, handler, err := rpc.StartHTTPEndpoint(endpoint, apis, modules, cors, vhosts)
	if err != nil {
		return err
	}
	n.log.Info("HTTP endpoint opened", "url", fmt.Sprintf("http://%s", endpoint), "cors", strings.Join(cors, ","), "vhosts", strings.Join(vhosts, ","))
	// All listeners booted successfully
	n.httpEndpoint = endpoint
	n.httpListener = listener
	n.httpHandler = handler

	return nil
}
```
startWS
```go
func (n *Node) startWS(endpoint string, apis []rpc.API, modules []string, wsOrigins []string, exposeAll bool) error {
	// Short circuit if the WS endpoint isn't being exposed
	if endpoint == "" {
		return nil
	}
	listener, handler, err := rpc.StartWSEndpoint(endpoint, apis, modules, wsOrigins, exposeAll)
	if err != nil {
		return err
	}
	n.log.Info("WebSocket endpoint opened", "url", fmt.Sprintf("ws://%s", listener.Addr()))
	// All listeners booted successfully
	n.wsEndpoint = endpoint
	n.wsListener = listener
	n.wsHandler = handler

	return nil
}
```

stack.Start()が終わったらシグナルを待ち受ける

```go
	go func() {
		sigc := make(chan os.Signal, 1)
		signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
		defer signal.Stop(sigc)
		<-sigc
		go stack.Stop()
		for i := 10; i > 0; i-- {
			<-sigc
			if i > 1 {
				log.Warn("Already shutting down, interrupt more to panic.", "times", i-1)
			}
		}
	}()
```

utils.StartNode(stack)が終わったら  
1番目のkeystoreのアカウントをunlockする

```go
	ks := stack.AccountManager().Backends(keystore.KeyStoreType)[0].(*keystore.KeyStore)

	passwords := utils.MakePasswordList(ctx)
	unlocks := strings.Split(ctx.GlobalString(utils.UnlockedAccountFlag.Name), ",")
	for i, account := range unlocks {
		if trimmed := strings.TrimSpace(account); trimmed != "" {
			unlockAccount(ctx, ks, trimmed, i, passwords)
		}
	}

```

AccountManagerがeventを待ち受ける

```go
	events := make(chan accounts.WalletEvent, 16)
	stack.AccountManager().Subscribe(events)

type WalletEventType int

const (
	// WalletArrived is fired when a new wallet is detected either via USB or via
	// a filesystem event in the keystore.
	WalletArrived WalletEventType = iota
	// WalletOpened is fired when a wallet is successfully opened with the purpose
	// of starting any background processes such as automatic key derivation.
	WalletOpened
	// WalletDropped
	WalletDropped
)

// WalletEvent is an event fired by an account backend when a wallet arrival or
// departure is detected.
type WalletEvent struct {
	Wallet Wallet          // Wallet instance arrived or departed
	Kind   WalletEventType // Event type that happened in the system
}
```

```go
	go func() {
		// Create an chain state reader for self-derivation
		rpcClient, err := stack.Attach()
		if err != nil {
			utils.Fatalf("Failed to attach to self: %v", err)
		}
		stateReader := ethclient.NewClient(rpcClient)

		// Open any wallets already attached
		for _, wallet := range stack.AccountManager().Wallets() {
			if err := wallet.Open(""); err != nil {
				log.Warn("Failed to open wallet", "url", wallet.URL(), "err", err)
			}
		}
		// Listen for wallet event till termination
		for event := range events {
			switch event.Kind {
			case accounts.WalletArrived:
				if err := event.Wallet.Open(""); err != nil {
					log.Warn("New wallet appeared, failed to open", "url", event.Wallet.URL(), "err", err)
				}
			case accounts.WalletOpened:
				status, _ := event.Wallet.Status()
				log.Info("New wallet appeared", "url", event.Wallet.URL(), "status", status)

				if event.Wallet.URL().Scheme == "ledger" {
					event.Wallet.SelfDerive(accounts.DefaultLedgerBaseDerivationPath, stateReader)
				} else {
					event.Wallet.SelfDerive(accounts.DefaultBaseDerivationPath, stateReader)
				}

			case accounts.WalletDropped:
				log.Info("Old wallet dropped", "url", event.Wallet.URL())
				event.Wallet.Close()
			}
		}
	}()

// Attach creates an RPC client attached to an in-process API handler.
// DialInProcでnet.pipe connectionを作成する
func (n *Node) Attach() (*rpc.Client, error) {
	n.lock.RLock()
	defer n.lock.RUnlock()

	if n.server == nil {
		return nil, ErrNodeStopped
	}
	return rpc.DialInProc(n.inprocHandler), nil
}

```

マイニングを始める
ethereum.StartMining(true)


## node.Wait()
stopをまつ


