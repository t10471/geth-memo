
## まとめ
makeFullNode

configNodeを作る

## config
```
	cfg := gethConfig{
		Eth:       eth.DefaultConfig,
		Shh:       whisper.DefaultConfig,
		Node:      defaultNodeConfig(),
		Dashboard: dashboard.DefaultConfig,
	}
```

## SetNodeConfig
```
	SetP2PConfig(ctx, &cfg.P2P)
	setIPC(ctx, cfg)
	setHTTP(ctx, cfg)
	setWS(ctx, cfg)
	setNodeUserIdent(ctx, cfg)
```

# SetP2PConfig
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

# setIPC
プライベートネットの場合に使う?
IPCPathを設定する

node.New

AccountManagerを作る
