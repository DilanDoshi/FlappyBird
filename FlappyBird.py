import neat.nn.feed_forward
import pygame, neat, time, os, random
pygame.font.init()

# Description: Object Oriented Flappy Bird set up with a NEAT (NeuroEvolution of Augmenting Topologies) algorithm 
#   that creates artifical neural networks in each invidual bird. The algorithm learns how to play the game succesfully
# 
# Note: NEAT Algorithm and config-feedforward implemented personally using algorithm and library documentation 
#   on NEAT website. Graphics completed with tutorial from YouTube.


WIN_WIDTH = 600
WIN_HEIGHT = 800


# Loading the images onto the screen 
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans",50)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count +=1
        d = self.vel*self.tick_count + 1.5*self.tick_count**2 #kinematic equation, provides bird with arc movement 

        #Fine-Tuning Movement 
        if d>=16: 
            d = 16
        if d < 0:
            d -=1 
        
        self.y = self.y + d

        if d<0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else: 
            if self.tilt > - 90:
                self.tilt -= self.ROT_VEL
    
    def draw(self, win):
        self.img_count += 1

        #Flapping animation as time progresses
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*4+1:
            self.img = self.IMGS[0]
            self.img_count = 0

        #If falling, do not flap wings
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
        
        #Rotating image
        rotated_image = pygame.transform.rotate(self.img,self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x,self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)    
class Pipe:
    GAP = 200
    VEL = 5

    def __init__ (self, x):
        self.x = x
        self.height = 0


        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG,False,True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(40,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL
    
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x,self.top))
        win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))
    
    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask,bottom_offset) #returns none if no collision
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True
        
        return False
class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__ (self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1+self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 +self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
        
    def draw(self, win):
        win.blit(self.IMG, (self.x1,self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG,(0,0))

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit (text, (WIN_WIDTH -10 - text.get_width(),10))

    for pipe in pipes:
        pipe.draw(win)
    
    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    birds = []
    nn = []
    g = []

    # Instatiating each bird with network and corresponding genome
    for genome_ID,genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nn.append(net)
        birds.append(Bird(230,350))
        g.append(genome)

    base = Base(730)
    pipes = [Pipe(700)]

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    run = True
    mult = 0
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        for i,bird in enumerate(birds):
            bird.move()
            g[i].fitness += 0.2
            mult +=0.01

            pipe_ind = 0

            #If bird has passed pipe, look at the next pipe
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipe.PIPE_TOP.get_width():
                pipe_ind = 1

            pipe = pipes[pipe_ind]
            if len(birds)<0:
                run = False
                break

            #NN input
            inputs = [
                bird.y,  # Bird's vertical position
                abs(bird.y - pipe.height),  # Distance to top of pipe
                abs(bird.y - pipe.bottom)   # Distance to bottom of pipe
            ]

            #NN output
            output = nn[i].activate(inputs)
            if output[0] > 2:
                bird.jump()


        base.move()
        add_pipe = False
        rem = []

        for pipe in pipes:
            for n,bird in enumerate(birds):
                if pipe.collide(bird,win):

                    # If bird collides, then penalize fitness and remove bird from list
                    g[n].fitness -= 1
                    birds.pop(n)
                    nn.pop(n)
                    g.pop(n)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

  
            pipe.move()
        if add_pipe:
            score +=1

            #Reward fitness for passing each pipe
            for genome in g:
                genome.fitness += 6

            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for n,bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < -10:
                # If bird hits floor or flies too high, penalize fitness and remove from list
                g[n].fitness -= 1

                birds.pop(n)
                nn.pop(n)
                g.pop(n)
        
        if not birds:
            break

        draw_window(win,birds,pipes,base,score)





def run(config_path):
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, 
                         neat.DefaultStagnation, 
                         config_path)
    
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal. From XOR example on documentation
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    #Decided not to have a reporter
    #p.add_reporter(neat.Checkpointer(5))

    winner = p.run(main, 30) #use main so that the birds can be modeled on the screen, run for up to 30 generations

if __name__ == '__main__':

    #Path to config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

