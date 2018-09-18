import pygame, time


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
print("number of joystick is ", joystick_count)

joystick = pygame.joystick.Joystick(0)
joystick.init()

while True:
    #for event in pygame.event.get(): # User did something
    #    pass

    print("a time")
    for i in range(15):
        if joystick.get_button(i):
            print(i)
    time.sleep(1)
    pygame.event.pump()

'''
done=False
clock = pygame.time.Clock()
while True: #game loop
    for event in pygame.event.get(): #loop through all the current events, such as key presses.
        if event.type == pygame.KEYDOWN:
            print("some key down")
            if event.key == pygame.K_ESCAPE: #it's better to have these as multiple statments in case you want to track more than one type of key press in the future.
                print("escape")
'''
