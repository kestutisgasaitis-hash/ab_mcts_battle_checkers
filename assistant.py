import pygame
import time
from threading import Thread
import copy
import math
import random

# ----- Konstantos -----
EMPTY, WHITE, BLACK, KING = 0, 1, 2, 4
WHITE_MAN, BLACK_MAN = WHITE, BLACK
WHITE_KING, BLACK_KING = WHITE | KING, BLACK | KING
SIZE, SQUARE_SIZE = 8, 70
BOARD_SIZE = SIZE * SQUARE_SIZE
STATUS_HEIGHT = 160 # Dar daugiau vietos gražiam UI
WINDOW_WIDTH, WINDOW_HEIGHT = BOARD_SIZE, BOARD_SIZE + STATUS_HEIGHT

# Spalvų paletė
WHITE_COLOR, BLACK_COLOR = (240, 217, 181), (118, 150, 86)
PIECE_WHITE, PIECE_BLACK, PIECE_KING = (255, 255, 255), (15, 15, 15), (255, 215, 0)
BG_DARK = (25, 25, 25)
ACCENT_BLUE = (0, 180, 255)
ACCENT_RED = (220, 50, 50)
ACCENT_GREEN = (50, 200, 50)

DIRS_WHITE = [(-1, -1), (-1, 1)]
DIRS_BLACK = [(1, -1), (1, 1)]
DIRS_KING = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# ----- Lentos logika (nepakito, nes veikia puikiai) -----
def initial_board():
    board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    for r in range(5, 8):
        for c in range(SIZE):
            if (r + c) & 1: board[r][c] = WHITE_MAN
    for r in range(3):
        for c in range(SIZE):
            if (r + c) & 1: board[r][c] = BLACK_MAN
    return board

def opponent(color): return BLACK if color == WHITE else WHITE
def is_king(piece): return piece & KING
def belongs_to(color, piece): return (piece & 3) == color
def in_bounds(r, c): return 0 <= r < SIZE and 0 <= c < SIZE
def copy_board(board): return [row[:] for row in board]

def promote_if_needed(r, piece):
    if piece == WHITE_MAN and r == 0: return WHITE_KING
    if piece == BLACK_MAN and r == SIZE - 1: return BLACK_KING
    return piece

def generate_moves(board, color):
    moves, captures = [], []
    for r in range(SIZE):
        for c in range(SIZE):
            p = board[r][c]
            if p != EMPTY and belongs_to(color, p):
                caps = find_captures_from(board, r, c, p)
                if caps: captures.extend(caps)
                elif not captures: moves.extend(find_simple_moves(board, r, c, p))
    return captures if captures else moves

def find_simple_moves(board, r, c, piece):
    moves = []
    dirs = DIRS_KING if is_king(piece) else (DIRS_WHITE if (piece & 3) == WHITE else DIRS_BLACK)
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and board[nr][nc] == EMPTY:
            moves.append((r, c, nr, nc, []))
    return moves

def find_captures_from(board, sr, sc, piece):
    res = []
    def dfs(r, c, p, captured_set):
        found = False
        dirs = DIRS_KING if is_king(p) else (DIRS_WHITE if (p & 3) == WHITE else DIRS_BLACK)
        for dr, dc in dirs:
            mr, mc, er, ec = r+dr, c+dc, r+2*dr, c+2*dc
            if in_bounds(er, ec) and board[er][ec] == EMPTY:
                mid = board[mr][mc]
                if mid != EMPTY and not belongs_to(p&3, mid) and (mr, mc) not in captured_set:
                    found = True
                    dfs(er, ec, promote_if_needed(er, p), captured_set | {(mr, mc)})
        if not found and captured_set:
            res.append((sr, sc, r, c, list(captured_set)))
    dfs(sr, sc, piece, set())
    return res

def make_move(board, sr, sc, er, ec, captured):
    p = board[sr][sc]
    board[sr][sc] = EMPTY
    board[er][ec] = promote_if_needed(er, p)
    for r, c in captured: board[r][c] = EMPTY

def evaluate(board, color):
    w_s, b_s = 0, 0
    for r in range(SIZE):
        for c in range(SIZE):
            p = board[r][c]
            if p == EMPTY: continue
            val = 4.5 if is_king(p) else 1.0
            if (p & 3) == WHITE:
                if not is_king(p): val += (7 - r) * 0.18
                w_s += val
            else:
                if not is_king(p): val += r * 0.18
                b_s += val
    return b_s - w_s if color == BLACK else w_s - b_s

# ----- Algoritmai -----
transposition_table = {}

def alpha_beta_search(board, depth, alpha, beta, color):
    
    board_key = (tuple(tuple(row) for row in board), color, depth)
    
    if board_key in transposition_table: 
        return transposition_table[board_key]
    
    moves = generate_moves(board, color)
    
    if depth == 0 or not moves: 
        return evaluate(board, color), None
    
    moves.sort(key=lambda m: len(m[4]), reverse=True)
    best_move, max_eval = moves[0], -float('inf')
    for move in moves:
        nb = copy_board(board); make_move(nb, *move[:4], move[4])
        ev = -alpha_beta_search(nb, depth-1, -beta, -alpha, opponent(color))[0]
        if ev > max_eval: max_eval, best_move = ev, move
        alpha = max(alpha, ev)
        if beta <= alpha: break
    transposition_table[board_key] = (max_eval, best_move)
    return max_eval, best_move

class MCTSNode:
    __slots__ = ['board', 'color', 'parent', 'move', 'children', 'wins', 'visits', 'untried_moves']
    def __init__(self, board, color, parent=None, move=None):
        self.board, self.color, self.parent, self.move = board, color, parent, move
        self.children, self.wins, self.visits = [], 0, 0
        self.untried_moves = generate_moves(board, color)
        random.shuffle(self.untried_moves)

def mcts_search(root_board, color, iterations=25000, time_limit=3.0):
    moves = generate_moves(root_board, color)
    if not moves:
        return 0.5, None
    if len(moves) == 1:
        return 0.5, moves[0]
    root = MCTSNode(copy_board(root_board), color)
    start = time.time()
    for i in range(iterations):
        if i & 127 == 0 and (time.time() - start) > time_limit: break
        node = root
        while not node.untried_moves and node.children:
            node = max(  node.children  , key=lambda c: (c.wins/(c.visits + 1e-8)+ 0.7 * math.sqrt(math.log(node.visits+1)/(c.visits + 1e-8)   ) )     )
        if node.untried_moves:
            m = node.untried_moves.pop()
            nb = copy_board(node.board); make_move(nb, *m[:4], m[4])
            child = MCTSNode(nb, opponent(node.color), node, m)
            node.children.append(child); node = child
        s_board, s_color = copy_board(node.board), node.color
        reward = 0.5
        for _ in range(55):
            m_s = generate_moves(s_board, s_color)
            if not m_s:
                reward = 1.0 if s_color == opponent(color) else 0.0; break
            make_move(s_board, *random.choice(m_s)[:4], random.choice(m_s)[4])
            s_color = opponent(s_color)
        else: 
            reward = 1 / (1 + math.exp(-0.4 * evaluate(s_board, color)))
        
        while node:
            node.visits += 1
            node.wins += reward if (node.parent and node.parent.color == color) else (1 - reward)
            node = node.parent
    best = max(root.children, key=lambda c: c.visits)
    return best.wins/best.visits, best.move

# ----- Gražus UI ir Žaidimo valdymas -----
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Grandmaster Battle: Alpha-Beta vs MCTS")
        self.font_main = pygame.font.Font(None, 28)
        self.font_bold = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 22)
        
        self.stats = {"MCTS": 0, "AB": 0, "Draws": 0, "Games": 0}
        self.reset_game()

    def reset_game(self):
        self.board = initial_board()
        self.current_color = WHITE
        self.no_capture_counter = 0
        self.move_count = 0
        self.ai_thinking = False
        self.win_rate = 0.5
        global transposition_table
        transposition_table = {}

    def draw_status_bar(self):
        # Fonas
        pygame.draw.rect(self.screen, BG_DARK, (0, BOARD_SIZE, WINDOW_WIDTH, STATUS_HEIGHT))
        pygame.draw.line(self.screen, (100, 100, 100), (0, BOARD_SIZE), (WINDOW_WIDTH, BOARD_SIZE), 2)

        # 1. Viršutinė info: Žaidimo numeris ir bendra statistika
        game_text = self.font_bold.render(f"GAME #{self.stats['Games'] + 1}", True, (255, 255, 255))
        self.screen.blit(game_text, (20, BOARD_SIZE + 15))
        
        stat_text = f"AB: {self.stats['AB']}  |  MCTS: {self.stats['MCTS']}  |  Draws: {self.stats['Draws']}"
        stat_render = self.font_main.render(stat_text, True, PIECE_KING)
        self.screen.blit(stat_render, (WINDOW_WIDTH - stat_render.get_width() - 20, BOARD_SIZE + 15))

        # 2. Vidurinė dalis: Kas dabar mąsto
        white_status = "AB (White) Thinking..." if (self.current_color == WHITE and self.ai_thinking) else "AB (White) Waiting"
        black_status = "MCTS (Black) Thinking..." if (self.current_color == BLACK and self.ai_thinking) else "MCTS (Black) Waiting"
        
        # Baltųjų indikatorius
        w_col = ACCENT_RED if (self.current_color == WHITE and self.ai_thinking) else (150, 150, 150)
        self.screen.blit(self.font_main.render(white_status, True, w_col), (20, BOARD_SIZE + 55))
        
        # Juodųjų indikatorius
        b_col = ACCENT_BLUE if (self.current_color == BLACK and self.ai_thinking) else (150, 150, 150)
        self.screen.blit(self.font_main.render(black_status, True, b_col), (WINDOW_WIDTH - 250, BOARD_SIZE + 55))

        # 3. Apatinė dalis: Win Probability Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, BOARD_SIZE + 100, WINDOW_WIDTH - 40, 25
        pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        
        # Užpildas (MCTS Win Rate)
        fill_w = int(bar_w * self.win_rate)
        pygame.draw.rect(self.screen, ACCENT_BLUE, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
        
        # Užrašas ant baro
        prob_text = f"Win Prob (MCTS): {self.win_rate*100:.1f}%"
        prob_render = self.font_small.render(prob_text, True, (255, 255, 255))
        self.screen.blit(prob_render, (bar_x + (bar_w // 2) - (prob_render.get_width() // 2), bar_y + 4))

        # 4. Papildoma informacija pačioje apačioje
        move_info = f"Moves without capture: {self.no_capture_counter} / 80"
        self.screen.blit(self.font_small.render(move_info, True, (120, 120, 120)), (20, BOARD_SIZE + 135))

    def draw_board(self):
        for r in range(SIZE):
            for c in range(SIZE):
                col = WHITE_COLOR if (r+c)%2==0 else BLACK_COLOR
                pygame.draw.rect(self.screen, col, (c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                p = self.board[r][c]
                if p != EMPTY:
                    pc = PIECE_WHITE if (p&3)==WHITE else PIECE_BLACK
                    pygame.draw.circle(self.screen, pc, (c*SQUARE_SIZE+35, r*SQUARE_SIZE+35), 28)
                    if is_king(p):
                        pygame.draw.circle(self.screen, PIECE_KING, (c*SQUARE_SIZE+35, r*SQUARE_SIZE+35), 10)
                        pygame.draw.circle(self.screen, (0,0,0), (c*SQUARE_SIZE+35, r*SQUARE_SIZE+35), 11, 2)

    def ai_step(self):
        if self.current_color == WHITE:
            # Alpha-Beta (White)
            _, move = alpha_beta_search(copy_board(self.board),6, -float('inf'), float('inf'), WHITE)
        else:
            # MCTS (Black)
            wr, move = mcts_search(copy_board(self.board), BLACK)
            self.win_rate = wr
            
        if move:
            if move[4]: self.no_capture_counter = 0
            else: self.no_capture_counter += 1
            make_move(self.board, *move[:4], move[4])
            self.move_count += 1
            
        self.current_color = opponent(self.current_color)
        self.ai_thinking = False

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return

            if not self.ai_thinking:
                moves = generate_moves(self.board, self.current_color)
                winner = None
                if not moves: winner = "AB" if self.current_color == BLACK else "MCTS"
                elif self.no_capture_counter >= 80: winner = "Draw"
                
                if winner:
                    self.stats["Games"] += 1
                    self.stats[winner if winner != "Draw" else "Draws"] += 1
                    print(f"Partija {self.stats['Games']}: {winner}")
                    time.sleep(1.0)
                    self.reset_game()
                    continue
                
                self.ai_thinking = True
                Thread(target=self.ai_step, daemon=True).start()

            self.screen.fill(BG_DARK)
            self.draw_board()
            self.draw_status_bar()
            pygame.display.flip()
            clock.tick(30)

if __name__ == "__main__":
    Game().run()
