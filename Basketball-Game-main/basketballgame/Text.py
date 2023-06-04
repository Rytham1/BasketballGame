import pygame

BLACK = (0, 0, 0)
RED = (255, 0, 0)


class Text:
    def text_objects(self, text, font, color):
        textSurface = font.render(text, True, color)
        return textSurface, textSurface.get_rect()

    def score_display(self, world, screen):
        p1color = RED if world.p1turn else BLACK
        p2color = BLACK if world.p1turn else RED
        basket = world.p1score + world.p2score
        self.add_to_screen(
            screen, 30, "Player 1: " + str(basket) + " points", 150, 50, p1color
        )
        #self.add_to_screen(
            #screen, 30, "Player 2: " + str(world.p2score) + " points", 150, 90, p2color
        #)

    def victory_message(self, world, screen):
        #winner = 1 if world.p1score > world.p2score else 2
        baskets = int((world.p1score + world.p2score)/10)
        self.add_to_screen(
            screen, 75, "Player 1 made " + str(baskets) + " baskets!", 640, 320,BLACK
        )

    def add_to_screen(self, screen, font_size, text, center_x, center_y, color):
        largeText = pygame.font.Font("freesansbold.ttf", font_size)
        TextSurf, TextRect = self.text_objects(text, largeText, color)
        TextRect.center = (center_x, center_y)
        screen.blit(TextSurf, TextRect)
