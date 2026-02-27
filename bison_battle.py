import pygame
import math
import random
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass

# --- KONFIGŪRACIJA ---
GAME_MODE = "BATTLE"  # "HUMAN_VS_MCTS" arba "BATTLE"
MCTS_ITERATIONS = 2000
AB_DEPTH = 8
WINDOW_WIDTH = 840  # 640 (lenta) + 200 (panelė)
WINDOW_HEIGHT = 640
# ---------------------

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

    def is_game_over_final(self): 
        # No legal moves = game over
        if len(self.generate_moves()) == 0:
            return True
        
        # 50-move rule: no captures or promotions = draw
        if self.moves_without_capture >= 60:
            return True
        
        return False   
                          
    def game_result(self): 
        # If 50-move rule triggered, it's a draw
        if self.moves_without_capture >= 30:
            return 0 #DRAW
        
        # Otherwise the side with no moves loses
        return -1 if self.side == WHITE else 1


# --- ALPHA-BETA VARIKLIS ---
class AlphaBetaPlayer:
    def __init__(self, depth=6):
        self.depth = depth
        self.nodes_visited = 0

    def search(self, position):
        self.nodes_visited = 0
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        moves = position.generate_moves()
        if not moves: return None
        
        # Rūšiuojame ėjimus (kirtimai pirma), kad Alpha-Beta būtų greitesnis
        moves.sort(key=lambda m: m.is_capture(), reverse=True)
        
        val, move = self._minimax(position, self.depth, alpha, beta, position.side == WHITE)
        return move, val

    def _minimax(self, pos, depth, alpha, beta, maximizing):
        self.nodes_visited += 1
        if depth == 0 or pos.is_game_over():
            return pos.evaluate(), None
            
        moves = pos.generate_moves()
        best_move = moves[0]
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                eval, _ = self._minimax(pos.make_move(move), depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                eval, _ = self._minimax(pos.make_move(move), depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval, best_move

# --- MCTS VARIKLIS (Su jūsų trupmeniniais krepšiniais) ---
class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.parent_action = action
        self.children = []
        self.n_visits = 0
        self.results = {WHITE: 0.0, BLACK: 0.0, DRAW: 0.0}
        self.untried = self.state.generate_moves()
        random.shuffle(self.untried)

    def is_fully_expanded(self): return len(self.untried) == 0

    def best_child(self, c=0.7):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.n_visits == 0: return child
            win_rate = (child.results[WHITE] if self.state.side == WHITE else child.results[BLACK]) / child.n_visits
            exploration = c * math.sqrt(math.log(self.n_visits) / child.n_visits)
            score = win_rate + exploration + (random.random() * 1e-6)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        action = self.untried.pop()
        next_state = self.state.make_move(action)
        child = MonteCarloTreeSearchNode(next_state, parent=self, action=action)
        self.children.append(child)
        return child

    def rollout(self):
        curr = self.state
        for _ in range(55):
            if curr.is_game_over(): break
            moves = curr.generate_moves()
            curr = curr.make_move(random.choice(moves))
        
        if curr.is_game_over():
            res = curr.game_result()
            return float(res)
            
        score = curr.evaluate()
        k = 0.4 # Jūsų parinktas optimalus k
        prob_white = 1 / (1 + math.exp(-k * score))
        return 2.0 * prob_white - 1.0

    def backpropagate(self, res):
        self.n_visits += 1
        if res > 0:
            self.results[WHITE] += res
            self.results[BLACK] += (1.0 - res)
        elif res < 0:
            self.results[BLACK] += abs(res)
            self.results[WHITE] += (1.0 - abs(res))
        else:
            self.results[DRAW] += 1.0
        if self.parent: self.parent.backpropagate(res)

def mcts_search(root_state, iterations=1000):
    root = MonteCarloTreeSearchNode(root_state)
    t0 = time.time()
    for _ in range(iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        if not node.is_fully_expanded():
            node = node.expand()
        result = node.rollout()
        node.backpropagate(result)
    
    if not root.children: return None, None
    best = max(root.children, key=lambda c: c.n_visits)
    stats = type('Stats', (), {})()
    stats.best_move = best.parent_action
    stats.time = time.time() - t0
    stats.n_visits = root.n_visits
    w_success = root.results[WHITE]
    b_success = root.results[BLACK]
    draws = root.results[DRAW]
    cur_success = w_success if root_state.side == WHITE else b_success
    stats.win_rate = (cur_success + 0.5 * draws) / root.n_visits
    return stats.best_move, stats

# --- GUI IR BATTLE LOGIKA ---
class CheckersGUI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 18)
        self.large_font = pygame.font.SysFont("Arial", 22, bold=True)
        self.pos = Position.starting_position()
        self.ab_player = AlphaBetaPlayer(depth=AB_DEPTH)
        self.last_stats = None
        self.last_ab_val = None
        self.last_time = 0
        self.move_count = 0

    def draw(self):
        self.screen.fill((40, 40, 40)) # Tamsus fonas panelei
        # Lentos piešimas (640x640)
        for r in range(8):
            for c in range(8):
                color = (222, 184, 135) if (r + c) % 2 == 0 else (139, 69, 19)
                pygame.draw.rect(self.screen, color, (c*80, r*80, 80, 80))
                sq = rc_to_sq(r, c)
                if sq != -1: self.draw_piece(sq, r, c)
        
        # Šoninės panelės piešimas
        self.draw_panel()
        pygame.display.flip()

    def draw_piece(self, sq, r, c):
        # ... (Jūsų standartinis figūrų piešimo kodas) ...
        center = (c*80 + 40, r*80 + 40)
        if self.pos.white_men & bit(sq):
            pygame.draw.circle(self.screen, (240, 240, 240), center, 30)
        elif self.pos.black_men & bit(sq):
            pygame.draw.circle(self.screen, (20, 20, 20), center, 30)
        if self.pos.white_kings & bit(sq):
            pygame.draw.circle(self.screen, (240, 240, 240), center, 30)
            pygame.draw.circle(self.screen, (255, 215, 0), center, 15)
        elif self.pos.black_kings & bit(sq):
            pygame.draw.circle(self.screen, (20, 20, 20), center, 30)
            pygame.draw.circle(self.screen, (255, 215, 0), center, 15)

    def draw_panel(self):
        x_offset = 650
        y = 20
        
        # Bendra info
        self.draw_text("BATTLE STATS", x_offset, y, (255, 255, 255), True)
        y += 40
        side_text = "WHITE (MCTS)" if self.pos.side == WHITE else "BLACK (AB)"
        self.draw_text(f"Turn: {side_text}", x_offset, y, (200, 200, 0))
        y += 30
        self.draw_text(f"Move: {self.move_count}", x_offset, y, (255, 255, 255))
        
        y += 60
        # MCTS Info
        self.draw_text("MCTS (White):", x_offset, y, (255, 255, 255), True)
        y += 25
        if self.last_stats:
            self.draw_text(f"Win Rate: {self.last_stats.win_rate:.1%}", x_offset, y, (100, 255, 100))
            y += 25
            self.draw_text(f"Iters: {self.last_stats.n_visits}", x_offset, y, (200, 200, 200))
            y += 25
            self.draw_text(f"Time: {self.last_stats.time:.2f}s", x_offset, y, (200, 200, 200))
        
        y += 60
        # Alpha-Beta Info
        self.draw_text("Alpha-Beta (Black):", x_offset, y, (255, 255, 255), True)
        y += 25
        if self.last_ab_val is not None:
            self.draw_text(f"Eval: {self.last_ab_val:+.2f}", x_offset, y, (100, 200, 255))
            y += 25
            self.draw_text(f"Depth: {AB_DEPTH}", x_offset, y, (200, 200, 200))
            y += 25
            self.draw_text(f"Time: {self.last_time:.2f}s", x_offset, y, (200, 200, 200))

    def draw_text(self, text, x, y, color, bold=False):
        surf = self.large_font.render(text, True, color) if bold else self.font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def run_battle(self):
        if self.pos.is_game_over() or self.ai_thinking:
            return

        self.ai_thinking = True
        self.draw() # Atnaujiname ekraną prieš pradedant skaičiuoti
        
        t_start = time.time()
        
        if self.pos.side == WHITE:
            # MCTS ėjimas
            move, stats = mcts_search(self.pos, iterations=MCTS_ITERATIONS)
            self.last_stats = stats
            if move:
                print(f"WHITE (MCTS) move: {to_notation(move.start)}-{to_notation(move.path[-1])} | WinRate: {stats.win_rate:.2%}")
        else:
            # Alpha-Beta ėjimas
            move, val = self.ab_player.search(self.pos)
            self.last_ab_val = val
            self.last_time = time.time() - t_start
            if move:
                print(f"BLACK (AB) move: {to_notation(move.start)}-{to_notation(move.path[-1])} | Eval: {val:.2f}")

        if move:
            self.pos = self.pos.make_move(move)
            self.move_count += 1
        
        self.ai_thinking = False
        self.draw()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("MCTS vs Alpha-Beta Battle")
    gui = CheckersGUI(screen)
    gui.ai_thinking = False # Pridedame šį kintamąjį
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Battle režimas: AI žaidžia automatiškai
        if GAME_MODE == "BATTLE" and not gui.pos.is_game_over_final() and not gui.ai_thinking:
            gui.run_battle()
            pygame.time.delay(200) # Nedidelė pauzė tarp ėjimų vizualizacijai
        
        gui.draw()
        clock.tick(30) # Ribojame FPS, kad procesorius nekaistų be reikalo
        
    pygame.quit()


if __name__ == "__main__":
    main()