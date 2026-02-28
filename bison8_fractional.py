
"""
Checkers game with Pygame GUI and MCTS AI.
Rules: English Draughts (Men capture forward only, Kings capture both ways).
Fixed: Proper draw evaluation for 1K vs 1K positions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame
import time
import math
import random

WHITE = 0
BLACK = 1
DRAW = 2


# Svorio masyvai paprastoms šaškėms (pažangos bonusas)
# Baltieji juda link r=0, Juodieji link r=7
WHITE_PAWN_WEIGHTS = [0.0] * 32
BLACK_PAWN_WEIGHTS = [0.0] * 32


# Lentos koordinačių susiejimas su 32 langelių indeksais
_SQ_TO_RC = [None] * 32
_RC_TO_SQ = [[-1] * 8 for _ in range(8)]

_sq = 0
for r in range(8):
    for c in range(8):
        if (r + c) % 2 == 1:
            _SQ_TO_RC[_sq] = (r, c)
            _RC_TO_SQ[r][c] = _sq
            _sq += 1

def rc_to_sq(r: int, c: int) -> int:
    if 0 <= r < 8 and 0 <= c < 8: return _RC_TO_SQ[r][c]
    return -1

def sq_to_rc(sq: int) -> Tuple[int, int]:
    return _SQ_TO_RC[sq]
    
for sq in range(32):
    r, _ = sq_to_rc(sq)
    # Baltieji: r=7 (0.0 bonusas), r=0 (0.7 bonusas)
    WHITE_PAWN_WEIGHTS[sq] = 1.0 + (7 - r) * 0.18
    # Juodieji: r=0 (0.0 bonusas), r=7 (0.7 bonusas)
    BLACK_PAWN_WEIGHTS[sq] = 1.0 + r * 0.18

    
    
    

# Kaimyninių langelių masyvai
NW, NE, SW, SE = ([-1] * 32 for _ in range(4))
for sq in range(32):
    r, c = sq_to_rc(sq)
    NW[sq], NE[sq] = rc_to_sq(r - 1, c - 1), rc_to_sq(r - 1, c + 1)
    SW[sq], SE[sq] = rc_to_sq(r + 1, c - 1), rc_to_sq(r + 1, c + 1)

WHITE_PROMOTION = set(i for i in range(32) if sq_to_rc(i)[0] == 0)
BLACK_PROMOTION = set(i for i in range(32) if sq_to_rc(i)[0] == 7)

def bit(sq: int) -> int: return 1 << sq

def to_notation(sq: int) -> int:
    return 32 - sq

@dataclass
class Move:
    start: int
    path: List[int]
    captured: List[int]
    promote: bool = False

    def is_capture(self) -> bool:
        return len(self.captured) > 0

    def __str__(self) -> str:
        sep = "x" if self.is_capture() else "-"
        return f"{to_notation(self.start)}{sep}{to_notation(self.path[-1])}" + (" (K)" if self.promote else "")

class Position:
    def __init__(self, white_men, white_kings, black_men, black_kings, side=WHITE, moves_without_capture=0):
        self.white_men, self.white_kings = white_men, white_kings
        self.black_men, self.black_kings = black_men, black_kings
        self.side = side
        self.moves_without_capture = moves_without_capture

    @staticmethod
    def starting_position():
        w, b = 0, 0
        for sq in range(32):
            r, _ = sq_to_rc(sq)
            if r <= 2: b |= bit(sq)
            elif r >= 5: w |= bit(sq)
        return Position(w, 0, b, 0, WHITE, 0)

    def occupied(self) -> int: return self.white_men | self.white_kings | self.black_men | self.black_kings
    def own_men(self) -> int: return self.white_men if self.side == WHITE else self.black_men
    def own_kings(self) -> int: return self.white_kings if self.side == WHITE else self.black_kings
    def opp_men(self) -> int: return self.black_men if self.side == WHITE else self.white_men
    def opp_kings(self) -> int: return self.black_kings if self.side == WHITE else self.white_kings

    def clone(self): return Position(self.white_men, self.white_kings, self.black_men, self.black_kings, self.side, self.moves_without_capture)


    def evaluate(self):
        """
        Optimizuotas vertinimas iš BALTŲJŲ perspektyvos.
        Naudoja bitboardus ir iš anksto paruoštus svorius.
        """
        score = 0.0
    
        # 1. Materialinis svoris (Karalius = 2.5 paprastos šaškės)
        # bit_count() yra labai greitas Python 3.10+
        w_pawns_bits = self.white_men & ~self.white_kings
        b_pawns_bits = self.black_men & ~self.black_kings
    
        score += w_pawns_bits.bit_count() * 1.0
        score += self.white_kings.bit_count() * 2.0
    
        score -= b_pawns_bits.bit_count() * 1.0
        score -= self.black_kings.bit_count() * 2.0

        # 2. Pozicinis bonusas už pažangą (naudojant jūsų paruoštus svorius)
        # Einame tik per nustatytus bitus (maksimaliai 12 per pusę)
    
        # Baltųjų pažanga (juda link r=0)
        temp_w = w_pawns_bits
        while temp_w:
            sq = (temp_w & -temp_w).bit_length() - 1
            # Pridedame tik bonuso dalį (atėmus bazinį 1.0)
            score += (WHITE_PAWN_WEIGHTS[sq] - 1.0)
            temp_w &= temp_w - 1

        # Juodųjų pažanga (juda link r=7)
        temp_b = b_pawns_bits
        while temp_b:
            sq = (temp_b & -temp_b).bit_length() - 1
            # Atimame tik bonuso dalį
            score -= (BLACK_PAWN_WEIGHTS[sq] - 1.0)
            temp_b &= temp_b - 1

        # 3. Saugus užnugaris (Back-rank bonusas)
        # Baltieji saugo r=7 (bitai 28-31), Juodieji saugo r=0 (bitai 0-3)
        # Kaukės: 0xF0000000 (r=7), 0x0000000F (r=0)
    
        white_back_rank = self.white_men & 0xF0000000
        black_back_rank = self.black_men & 0x0000000F
    
        score += white_back_rank.bit_count() * 0.1
        score -= black_back_rank.bit_count() * 0.1

        return score


    def make_move(self, move: Move) -> "Position":
        new_pos = self.clone()
        f, t = move.start, move.path[-1]
        is_w = (self.white_men | self.white_kings) & bit(f)
        is_k = (self.white_kings | self.black_kings) & bit(f)
        new_pos.white_men &= ~bit(f); new_pos.white_kings &= ~bit(f)
        new_pos.black_men &= ~bit(f); new_pos.black_kings &= ~bit(f)
        for csq in move.captured:
            new_pos.white_men &= ~bit(csq); new_pos.white_kings &= ~bit(csq)
            new_pos.black_men &= ~bit(csq); new_pos.black_kings &= ~bit(csq)
        wbk = is_k or move.promote
        if is_w:
            if wbk: new_pos.white_kings |= bit(t)
            else: new_pos.white_men |= bit(t)
        else:
            if wbk: new_pos.black_kings |= bit(t)
            else: new_pos.black_men |= bit(t)
        new_pos.moves_without_capture = 0 if (move.is_capture() or (not is_k and move.promote)) else self.moves_without_capture + 1
        new_pos.side = BLACK if self.side == WHITE else WHITE
        return new_pos

    def generate_moves(self) -> List[Move]:
        caps = self._generate_captures()
        return caps if caps else self._generate_quiet_moves()

    def _generate_quiet_moves(self) -> List[Move]:
        occ, moves = self.occupied(), []
        m, dirs = self.own_men(), ([NW, NE] if self.side == WHITE else [SW, SE])
        while m:
            sq = (m & -m).bit_length() - 1; m &= m - 1
            for D in dirs:
                to = D[sq]
                if to != -1 and not ((occ >> to) & 1):
                    p = (to in (WHITE_PROMOTION if self.side == WHITE else BLACK_PROMOTION))
                    moves.append(Move(sq, [to], [], p))
        k = self.own_kings()
        while k:
            sq = (k & -k).bit_length() - 1; k &= k - 1
            for D in (NW, NE, SW, SE):
                to = D[sq]
                if to != -1 and not ((occ >> to) & 1): moves.append(Move(sq, [to], [], False))
        return moves

    def _generate_captures(self) -> List[Move]:
        opp, occ, captures = self.opp_men() | self.opp_kings(), self.occupied(), []
        
        def recurse(s_sq, c_sq, is_k, c_occ, c_opp, path, caps):
            res = []
            # Taisyklė: Paprastos šaškės kerta tik į priekį, Karaliai - visur
            if is_k:
                valid_dirs = (NW, NE, SW, SE)
            else:
                valid_dirs = (NW, NE) if self.side == WHITE else (SW, SE)
                
            for D in valid_dirs:
                mid = D[c_sq]
                if mid == -1 or not ((c_opp >> mid) & 1): continue
                land = D[mid]
                if land != -1 and (not ((c_occ >> land) & 1) or land == s_sq):
                    if mid in caps: continue
                    np, nc = path + [land], caps + [mid]
                    prom = (not is_k and ((self.side == WHITE and land in WHITE_PROMOTION) or (self.side == BLACK and land in BLACK_PROMOTION)))
                    if prom: res.append(Move(s_sq, np, nc, True))
                    else:
                        further = recurse(s_sq, land, is_k, c_occ | bit(land), c_opp & ~bit(mid), np, nc)
                        if further: res.extend(further)
                        else: res.append(Move(s_sq, np, nc, False))
            return res

        for p_type, is_k in [(self.own_men(), False), (self.own_kings(), True)]:
            p = p_type
            while p:
                sq = (p & -p).bit_length() - 1; p &= p - 1
                captures.extend(recurse(sq, sq, is_k, occ & ~bit(sq), opp, [], []))
        return captures

    def is_game_over(self): 
        # No legal moves = game over
        if len(self.generate_moves()) == 0:
            return True
        
        # 50-move rule: no captures or promotions = draw
        if self.moves_without_capture >= 30:
            return True
        
        return False
    
    def game_result(self): 
        # If 50-move rule triggered, it's a draw
        if self.moves_without_capture >= 30:
            return 0
        
        # Otherwise the side with no moves loses
        return -1 if self.side == WHITE else 1





# --- MCTS MAZGAS SU TRUPMENINIAIS KREPŠELIAIS ---

class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.parent_action = action
        self.children = []
        self.n_visits = 0
        
        # CRYSTAL CLEAR: Trys krepšeliai, saugantys trupmeninę statistiką
        self.results = {WHITE: 0.0, BLACK: 0.0, DRAW: 0.0}
        
        self.untried = self.state.generate_moves()
        # Atsitiktinis eiliškumas plėtrai
        random.shuffle(self.untried)

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def best_child(self, c=1.2):
        """
        Parenka geriausią vaiką pagal UCB1.
        Naudoja trupmeninę statistiką iš krepšelių.
        """
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            if child.n_visits == 0: return child
            
            # win_rate skaičiuojamas iš TĖVO perspektyvos
            # Kadangi krepšeliuose saugome trupmenas, tai yra vidutinė sėkmės tikimybė
            win_rate = (child.results[WHITE] if self.state.side == WHITE else child.results[BLACK]) / child.n_visits
            
            # UCB1 formulė su mikroskopiniu triukšmu tie-breaking'ui
            exploration = c * math.sqrt(math.log(self.n_visits) / child.n_visits)
            score = win_rate + exploration + (random.random() * 1e-6)
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self):
        action = self.untried.pop() # Greitas pop() dėl shuffle init'e
        next_state = self.state.make_move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        """
        Simuliacija su euristiniu vertinimu ir sigmoid transformacija.
        """
        curr = self.state
        # Simuliuojame iki 50 ėjimų
        for _ in range(50):
            if curr.is_game_over():
                break
            moves = curr.generate_moves()
            curr = curr.make_move(random.choice(moves))
        
        # Jei žaidimas baigėsi natūraliai (pvz. kirtimų nėra arba 50-move rule)
        if curr.is_game_over():
            res = curr.game_result() # 1 (White), -1 (Black), 0 (Draw)
            return float(res)
            
        # Jei simuliacija sustojo, vertiname poziciją
        score = curr.evaluate() # Iš BALTŲJŲ perspektyvos
        
        # Sigmoid: paverčiame score į tikimybę [0, 1]
        k = 0.4
        prob_white = 1 / (1 + math.exp(-k * score))
        
        # Grąžiname vertę skalėje [-1, 1]
        return 2.0 * prob_white - 1.0

    def backpropagate(self, res):
        """
        Paskirsto trupmeninį rezultatą į krepšelius.
        """
        self.n_visits += 1
        
        # res yra tarp -1.0 (Black) ir 1.0 (White)
        if res > 0: # Link baltųjų pergalės
            self.results[WHITE] += res
            self.results[BLACK] += (1.0 - res)
        elif res < 0: # Link juodųjų pergalės
            self.results[BLACK] += abs(res)
            self.results[WHITE] += (1.0 - abs(res))
        else: # Tikros lygiosios
            self.results[DRAW] += 1.0
            
        if self.parent:
            self.parent.backpropagate(res)

# --- PAGRINDINĖ PAIEŠKOS FUNKCIJA ---
def mcts_search(root_state, iterations=800):
    root = MonteCarloTreeSearchNode(root_state)
    t0 = time.time()
    
    for _ in range(iterations):
        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            
        # 2. Expansion
        if not node.is_fully_expanded():
            node = node.expand()
            
        # 3. Simulation
        result = node.rollout()
        
        # 4. Backpropagation
        node.backpropagate(result)
 
    # Rezultatų surinkimas
    if not root.children: return None, None
    
    # Robust Child: pasirenkame lankomiausią
    best = max(root.children, key=lambda c: c.n_visits)
    
    stats = type('Stats', (), {})()
    stats.best_move = best.parent_action
    stats.time = time.time() - t0
    stats.n_visits = root.n_visits
    
    # Win rate skaičiavimas iš dabartinio žaidėjo perspektyvos
    w_success = root.results[WHITE]
    b_success = root.results[BLACK]
    draws = root.results[DRAW]
    
    cur_success = w_success if root_state.side == WHITE else b_success
    stats.win_rate = (cur_success + 0.5 * draws) / root.n_visits
    
    return stats.best_move, stats


  
class CheckersGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 684))
        pygame.display.set_caption("Checkers AI - Forward Captures Only")
        self.font = pygame.font.Font(None, 26)
        self.num_font = pygame.font.Font(None, 20)
        self.position = Position.starting_position()
        self.selected_sq = None
        self.legal_moves = []
        self.last_info = "Jūsų eilė (Baltieji)"

    def draw(self):
        for r in range(8):
            for c in range(8):
                color = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)
                pygame.draw.rect(self.screen, color, (c*80, r*80, 80, 80))
                sq = rc_to_sq(r, c)
                if sq != -1:
                    num_txt = self.num_font.render(str(to_notation(sq)), True, (100, 70, 40))
                    self.screen.blit(num_txt, (c*80 + 5, r*80 + 5))

        if self.selected_sq is not None:
            r, c = sq_to_rc(self.selected_sq)
            pygame.draw.rect(self.screen, (0, 255, 0), (c*80, r*80, 80, 80), 4)
            for m in self.legal_moves:
                r, c = sq_to_rc(m.path[-1])
                s = pygame.Surface((80, 80), pygame.SRCALPHA)
                s.fill((255, 255, 0, 120))
                self.screen.blit(s, (c*80, r*80))

        for sq in range(32):
            r, c = sq_to_rc(sq); cx, cy = c*80 + 40, r*80 + 40
            if (self.position.white_men >> sq) & 1: pygame.draw.circle(self.screen, (255,255,255), (cx,cy), 30)
            elif (self.position.white_kings >> sq) & 1:
                pygame.draw.circle(self.screen, (255,255,255), (cx,cy), 30)
                pygame.draw.circle(self.screen, (255,215,0), (cx,cy), 12)
            elif (self.position.black_men >> sq) & 1: pygame.draw.circle(self.screen, (50,50,50), (cx,cy), 30)
            elif (self.position.black_kings >> sq) & 1:
                pygame.draw.circle(self.screen, (50,50,50), (cx,cy), 30)
                pygame.draw.circle(self.screen, (255,215,0), (cx,cy), 12)

        pygame.draw.rect(self.screen, (220,220,220), (0, 640, 640, 44))
        txt = self.font.render(self.last_info, True, (0,0,0))
        self.screen.blit(txt, (20, 655))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
                if event.type == pygame.MOUSEBUTTONDOWN and self.position.side == WHITE:
                    col, row = event.pos[0]//80, event.pos[1]//80
                    sq = rc_to_sq(row, col)
                    if sq != -1:
                        if (self.position.own_men() | self.position.own_kings()) & bit(sq):
                            self.selected_sq, self.legal_moves = sq, [m for m in self.position.generate_moves() if m.start == sq]
                        elif self.selected_sq is not None:
                            target_move = next((m for m in self.legal_moves if m.path[-1] == sq), None)
                            if target_move:
                                self.position = self.position.make_move(target_move)
                                self.selected_sq, self.legal_moves = None, []
            self.draw()
            pygame.display.flip()
            if self.position.side == BLACK and not self.position.is_game_over():
                self.last_info = "AI mąsto..."
                self.draw(); pygame.display.flip()
                move, stats = mcts_search(self.position)
                if move:
                    self.position = self.position.make_move(move)
                    self.last_info = f"AI: {move} | Win: {stats.win_rate*100:.1f}% | Laikas: {stats.time:.2f}s"
                #if self.position.is_game_over(): self.last_info = "Žaidimas baigtas!"


if __name__ == "__main__":
    CheckersGUI().run()
        
        
