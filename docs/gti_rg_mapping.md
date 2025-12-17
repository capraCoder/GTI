# GTI ↔ Robinson-Goforth Topology Mapping
# For academic validation and citation

## Quick Reference: 12 Symmetric Ordinal 2×2 Games

| R-G Layer | GTI Name | R-G Index | Equilibria | Dom.Strat | Social Dilemma | Status |
|-----------|----------|-----------|------------|-----------|----------------|--------|
| **L1: Discord** | Prisoner's Dilemma | 111 | 1 | 2 | Yes | ✓ Covered |
| L1 | Deadlock | 211 | 1 | 2 | No | ✓ Covered |
| L1 | Hero | 121 | 1 | 1 | — | ✗ Not in GTI |
| L1 | Leader | 112 | 1 | 1 | — | ✗ Not in GTI |
| **L2: Anti-coord** | Chicken | 222 | 2 | 0 | Yes | ✓ Covered |
| L2 | Battle of Sexes | 312 | 2 | 0 | No | ✓ Covered |
| **L3: Coord** | Stag Hunt | 311 | 2 | 0 | Yes | ✓ Covered |
| L3 | Assurance | 322 | 2 | 0 | Yes | ✓ Covered |
| L3 | Compromise | 331 | 1 | 0 | — | ✗ Not in GTI |
| **L4: Harmony** | Coordination | 411 | 2 | 0 | No | ✓ Covered |
| L4 | Harmony | 444 | 1 | 2 | No | ✓ Covered |
| L4 | Concord | 433 | 1 | 1 | — | ✗ Not in GTI |

**Coverage: 8/12 symmetric games (67%)**
**Missing: Hero, Leader, Compromise, Concord (low practical importance)**

---

## Layer Characteristics (Robinson-Goforth 2005)

| Layer | Name | Best Payoffs | Conflict Type | Example |
|-------|------|--------------|---------------|---------|
| 1 | Discord | Diagonally opposite | Mixed-motive, defection pressure | PD, Deadlock |
| 2 | Mixed | Off-diagonal | Anti-coordination | Chicken, BoS |
| 3 | Coordination | Aligned (partial) | Trust/assurance | Stag Hunt |
| 4 | Harmony | Fully aligned | No conflict | Pure Coordination |

---

## GTI Extensions Beyond R-G

R-G topology covers ONLY 2×2 simultaneous ordinal games.
GTI extends to:

| Category | Games | R-G Status |
|----------|-------|------------|
| Sequential | Ultimatum, Trust, Stackelberg, Centipede, Entry, Dictator | Not covered |
| Incomplete Info | Signaling, Screening | Not covered |
| N-player | Public Goods, Commons, Volunteer | 2-player only |
| Zero-sum | Matching Pennies | Cyclic (no pure eq) |

---

## Academic Citation Format

For GTI paper, cite as:

> The Game Theoretic Index (GTI) classification encompasses 8 of the 12 
> symmetric ordinal 2×2 games identified in the Robinson-Goforth topology 
> (Robinson & Goforth, 2005), covering all major social dilemmas 
> (Prisoner's Dilemma, Chicken, Stag Hunt) and coordination games. 
> GTI extends beyond the 2×2 framework to include sequential games, 
> incomplete information scenarios, and n-player generalizations.

**Key Reference:**
Robinson, D., & Goforth, D. (2005). *The Topology of 2×2 Games: A New Periodic Table*. 
London: Routledge.

**Supporting:**
Bruns, B. (2015). Names for Games: Locating 2×2 Games. *Games*, 6(4), 495-520.
Rapoport, A., & Guyer, M. (1966). A taxonomy of 2×2 games. *General Systems*, 11, 203-214.

---

## Validation Path

1. ✓ Map GTI types to R-G indices (this document)
2. ○ Cross-reference with games2p2k dataset (2,416 experimental cases)
3. ○ Validate classifier against R-G ground truth for symmetric games
4. ○ Document extensions (sequential, n-player) as GTI contribution
