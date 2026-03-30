[日本語版](../../ja/04-web-and-network/network-fundamentals/SKILL.md)

# Network Fundamentals — Complete Guide

> A systematic understanding of the protocol stack that powers the Internet. Covering everything from TCP/IP, HTTP, DNS, and TLS to WebSocket and gRPC — all the networking knowledge an engineer needs.

## Target Audience

- Engineers who want to learn networking from the ground up
- Web developers who need a deep understanding of HTTP/HTTPS
- Developers building with security and performance in mind

## Prerequisites

- Computer basics (bits, bytes, binary)
- Reference: [CS Fundamentals](../../01-cs-fundamentals/computer-science-fundamentals/)

## Guide Index

### 00-introduction (Introduction)
| File | Topic | Overview |
|------|-------|----------|
| [00-how-internet-works.md](docs/00-introduction/00-how-internet-works.md) | How the Internet Works | Packet switching, ISPs, submarine cables, routing |
| [01-osi-and-tcpip-model.md](docs/00-introduction/01-osi-and-tcpip-model.md) | OSI / TCP/IP Model | 7-layer / 4-layer models, role of each layer, protocol mapping |
| [02-ip-addressing.md](docs/00-introduction/02-ip-addressing.md) | IP Addressing | IPv4/IPv6, subnets, CIDR, NAT, DHCP |
| [03-dns.md](docs/00-introduction/03-dns.md) | DNS | Name resolution mechanics, recursive/iterative queries, DNS records |

### 01-protocols (Protocols)
| File | Topic | Overview |
|------|-------|----------|
| [00-tcp.md](docs/01-protocols/00-tcp.md) | TCP | 3-way handshake, flow control, congestion control |
| [01-udp.md](docs/01-protocols/01-udp.md) | UDP | Datagrams, real-time communication, QUIC |
| [02-websocket.md](docs/01-protocols/02-websocket.md) | WebSocket | Bidirectional communication, handshake, implementation patterns |
| [03-grpc.md](docs/01-protocols/03-grpc.md) | gRPC | Protocol Buffers, streaming, comparison with REST |

### 02-http (HTTP)
| File | Topic | Overview |
|------|-------|----------|
| [00-http-basics.md](docs/02-http/00-http-basics.md) | HTTP Basics | Methods, status codes, headers |
| [01-http2-and-http3.md](docs/02-http/01-http2-and-http3.md) | HTTP/2 and HTTP/3 | Multiplexing, server push, QUIC |
| [02-rest-api.md](docs/02-http/02-rest-api.md) | REST API Design | Resource design, versioning, HATEOAS |
| [03-caching.md](docs/02-http/03-caching.md) | HTTP Caching | Cache-Control, ETag, CDN |
| [04-cors.md](docs/02-http/04-cors.md) | CORS | Same-origin policy, preflight requests, configuration |

### 03-security (Security)
| File | Topic | Overview |
|------|-------|----------|
| [00-tls-ssl.md](docs/03-security/00-tls-ssl.md) | TLS/SSL | Handshake, certificates, cipher suites |
| [01-authentication.md](docs/03-security/01-authentication.md) | Authentication Methods | Basic, Bearer, OAuth 2.0, JWT |
| [02-common-attacks.md](docs/03-security/02-common-attacks.md) | Network Attacks | MITM, DNS poisoning, DDoS, countermeasures |

### 04-advanced (Advanced Topics)
| File | Topic | Overview |
|------|-------|----------|
| [00-load-balancing.md](docs/04-advanced/00-load-balancing.md) | Load Balancing | L4/L7, algorithms, health checks |
| [01-cdn.md](docs/04-advanced/01-cdn.md) | CDN | Edge caching, delivery optimization, configuration |
| [02-network-debugging.md](docs/04-advanced/02-network-debugging.md) | Network Debugging | curl, tcpdump, Wireshark, Chrome DevTools |
| [03-performance.md](docs/04-advanced/03-performance.md) | Network Optimization | Latency reduction, bandwidth optimization, connection management |

## Learning Path

```
Basics:      00-introduction → 01-protocols/00-01
Web:         02-http → 01-protocols/02-03
Security:    03-security
Operations:  04-advanced
```

## FAQ

### Q1: What order should I follow when learning networking?
Start with 00-introduction (how the Internet works, OSI/TCP/IP model, IP addressing, DNS) to build a solid foundation. Then move to 01-protocols (TCP, UDP) to understand the transport layer. From there, study 02-http (HTTP basics, HTTP/2 and HTTP/3, REST API, caching, CORS), followed by 03-security (TLS, authentication, attack vectors) for security knowledge, and finally 04-advanced (load balancing, CDN, debugging, performance) for operational skills.

### Q2: Do web developers really need deep networking knowledge?
Absolutely. The majority of performance issues are rooted in networking. For example, misconfigured HTTP caching, CORS errors, TLS handshake latency, and slow DNS resolution are problems you cannot even diagnose without a solid understanding of networking fundamentals. Additionally, in microservices architectures, network knowledge is essential for making informed decisions about technologies like gRPC and WebSocket.

### Q3: How does this Skill differ from the "Browser and Web Platform" Skill?
This Skill focuses on protocols and infrastructure (TCP/IP, HTTP, DNS, TLS, routing, etc.). The "Browser and Web Platform" Skill covers client-side technologies such as the browser rendering engine, DOM, Web APIs, and ServiceWorker. The two are complementary — studying both is recommended for a complete understanding of how the web works.

## Summary

This Skill provides a systematic treatment of the following topics:

- How every layer works, from the physical infrastructure of the Internet to the application layer (OSI model, TCP/IP, routing, DNS)
- Design philosophy, internal mechanics, and trade-offs of major protocols including TCP, UDP, WebSocket, and gRPC
- The evolution from HTTP/1.1 to HTTP/3, REST API design, caching strategies, and CORS
- Security knowledge covering TLS/SSL, authentication methods, network attacks and their countermeasures
- Practical skills in load balancing, CDN, network debugging, and performance optimization

## Related Skills

- [CS Fundamentals](../../01-cs-fundamentals/computer-science-fundamentals/) — Computer Science Fundamentals
- [Browser and Web Platform](../browser-and-web-platform/) — Browser and Web Platform
- [Security Fundamentals](../../06-data-and-security/security-fundamentals/) — Security Fundamentals

## References

- [Computer Networking: A Top-Down Approach](https://gaia.cs.umass.edu/kurose_ross/index.php) - By Kurose & Ross. The definitive textbook on network engineering, systematically covering each layer with a top-down approach
- [MDN Web Docs - HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP) - Mozilla's comprehensive HTTP reference covering headers, status codes, caching, and more
- [RFC Editor](https://www.rfc-editor.org/) - The official repository of Internet standards (RFCs). The authoritative source for all protocol specifications including TCP, UDP, HTTP, and TLS
- [High Performance Browser Networking](https://hpbn.co/) - By Ilya Grigorik. A practical guide to browser networking with in-depth coverage of TCP, TLS, HTTP/2, WebSocket, and more
- [Cloudflare Learning Center](https://www.cloudflare.com/learning/) - An accessible learning resource covering DNS, CDN, DDoS, TLS, and other networking technologies
