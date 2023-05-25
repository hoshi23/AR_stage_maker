import cv2
import random
import time
import numpy as np

G = 10  # gravity
USAGE = "left: J  right: K  jump: I  quit: Q"
IMAGES = {
    "human": "image/walk_boy_run.png",
    "flag": "image/undoukai_flag1.png",
    "coin": "image/money_kasoutsuuka_blank.png",
    "monster": "image/fantasy_behemoth.png",
}


class Object:
    direction: bool = True  # 初期値の方向: True, 逆向き: False
    width: int
    height: int
    x: int
    y: int
    vx: int = 0
    vy: int = 0
    dx: int = 20
    is_enable: bool = True

    def __init__(self, img_filepath: str, ratio: float = 1.0) -> None:
        original_img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(original_img, (int(original_img.shape[1]*ratio), int(original_img.shape[0]*ratio)))
        self.height, self.width = self.img.shape[:2]
        self.__set_mask()
        self.img = self.img[:, :, :3]

    def __set_mask(self):
        mask = self.img[:, :, 3]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.mask = mask / 255.0

    def flip_direction(self, direction: bool = True):
        if self.direction == direction:
            return
        self.img = cv2.flip(self.img, 1)
        self.mask = cv2.flip(self.mask, 1)
        self.direction = direction

    def set_object(self, x: int | None = None, y: int | None = None, vx: int | None = None, vy: int | None = None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if vx is not None:
            self.vx = vx
            if self.vx < 0:
                self.flip_direction(direction=True)
            elif self.vx > 0:
                self.flip_direction(direction=False)
        if vy is not None:
            self.vy = vy

    def update_object(self, height_max: int, width_max: int, dvx: int = 0, dvy: int = 0):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0:
            self.x = 0
        elif self.x > width_max - self.width:
            self.x = width_max - self.width
        if self.y < 0:
            self.y = 0
        elif self.y > height_max - self.height:
            self.y = height_max - self.height
        self.vx += dvx
        self.vy += dvy

    def delete(self):
        self.is_enable = False


class Game:
    ##################背景差分#########################
    # self.diff_model = cv2.bgsegm.createBackgroundSubtractorGMG()
    diff_model = cv2.createBackgroundSubtractorMOG2()
    # self.diff_model = cv2.bgsegm.createBackgroundSubtractorMOG()
    ##################################################

    human: Object = Object(img_filepath=IMAGES["human"], ratio=0.30)
    flag: Object = Object(img_filepath=IMAGES["flag"], ratio=0.30*1.5)
    monster: Object = Object(img_filepath=IMAGES["monster"], ratio=0.30)
    coins: list[Object] = [Object(img_filepath=IMAGES["coin"], ratio=0.3*0.5) for _ in range(10)]
    is_clear: bool = False
    is_gameover: bool = False
    score: int = 0

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.capture = cv2.VideoCapture(0)
        self.ret, self.first_frame = self.capture.read()
        while True:
            try:
                self.height, self.width = self.first_frame.shape[:2]
                break
            except AttributeError:
                time.sleep(1.0)
        self.mask = self.diff_model.apply(self.first_frame)
        self.human.set_object(x=0, y=self.height-self.human.height)
        self.flag.set_object(x=self.width//2, y=self.height//10, vx=0, vy=0)
        self.monster.set_object(x=self.width-self.monster.width, y=self.height//2-self.monster.height, vx=-7)
        self.__set_coins()

    def __set_coins(self):
        for coin in self.coins:
            coin.set_object(x=random.randint(0, self.width-coin.width), y=random.randint(0, self.height-coin.height))

    def reset(self):
        self.human.set_object(x=0, y=self.height-self.human.height, vx=0, vy=0)
        self.is_clear = False
        self.is_gameover = False
        self.__set_coins()
        self.score = 0


    def wait_key(self) -> bool:
        key = cv2.waitKey(50) & 0xff
        if key == ord('q'):
            return False
        if key == ord("i"):
            if self.human.y == self.height-self.human.height or 255 in self.mask[min(self.human.y+self.human.height//10*8,self.height-self.human.height):min(self.human.y+10+self.human.height,self.height-self.human.height), self.human.x:self.human.x+self.human.width]:
                self.human.set_object(vy=-self.human.height//2 - 5)
        if key == ord("j"):
            self.human.set_object(vx=-self.human.dx)
        if key == ord("k"):
            self.human.set_object(vx=self.human.dx)
        if key == ord("m"):
            self.human.vy = 0
            self.human.y += 20
            if self.human.y >= self.height-self.human.height:
                self.y = self.height-self.human.height
        if key == ord("r") and (self.is_clear or self.is_gameover):
            self.reset()
        return True

    def __display_object(self, obj: Object):
        if obj.is_enable:
            self.frame[obj.y:obj.y+obj.height, obj.x:obj.x+obj.width] = (1-obj.mask)*self.frame[obj.y:obj.y+obj.height, obj.x:obj.x+obj.width]
            self.frame[obj.y:obj.y+obj.height, obj.x:obj.x+obj.width] = obj.img*obj.mask + self.frame[obj.y:obj.y+obj.height, obj.x:obj.x+obj.width]

    def __display_maskedgh(self):
        if self.debug:
            if not self.is_clear:
                contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                areaframe = cv2.drawContours(self.frame, contours, -1, (255, 100, 0), 2)

    def __control_mask(self):
        mask = self.diff_model.apply(self.frame)
        ##################平滑化##########################
        # mask = cv2.blur(mask, (5,5))
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        ##################################################
        mask = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY)[1]
        mask_ratio = float(np.count_nonzero(mask == 255)) / (self.width * self.height)
        if mask_ratio < 0.3 and mask_ratio > 0.01:
            self.mask = mask

    def display_all_object(self):
        self.__display_object(self.human)
        self.__display_object(self.monster)
        self.__display_object(self.flag)
        for coin in self.coins:
            self.__display_object(coin)

    def judge_touch_object(self, obj1: Object, obj2: Object) -> bool:
        if (obj2.x >= obj1.x and obj2.x <= obj1.x+obj1.width) and (obj2.y >= obj1.y and obj2.y <= obj1.y+obj1.height):
            return True
        if (obj2.x+obj2.width >= obj1.x and obj2.x+obj2.width <= obj1.x+obj1.width) and (obj2.y >= obj1.y and obj2.y <= obj1.y+obj1.height):
            return True
        if (obj2.x >= obj1.x and obj2.x <= obj1.x+obj1.width) and (obj2.y+obj2.height >= obj1.y and obj2.y+obj2.height <= obj1.y+obj1.height):
            return True
        if (obj2.x+obj2.width >= obj1.x and obj2.x+obj2.width <= obj1.x+obj1.width) and (obj2.y+obj2.height >= obj1.y and obj2.y+obj2.height <= obj1.y+obj1.height):
            return True
        return False

    def run(self) -> bool:
        self.human.set_object(vx=0)
        self.ret, self.frame = self.capture.read()
        if not self.ret:
            return False
        self.__control_mask()
        if not self.wait_key():
            return False
        if self.human.vx >= 0 and 255 in self.mask[self.human.y:self.human.y+self.human.height//2, min(self.human.x+self.human.width,self.width-self.human.width):min(self.human.x+self.human.vx+self.human.width,self.width-self.human.width)]:
            self.human.set_object(vx=0)
        elif self.human.vx < 0 and 255 in self.mask[self.human.y:self.human.y+self.human.height//2, max(0, self.human.x+self.human.vx):self.human.x]:
            self.human.set_object(vx=0)
        if self.human.vy > 0 and 255 in self.mask[min(self.human.y+self.human.height//10*8,self.height-self.human.height):min(self.human.y+self.human.vy+self.human.height,self.height-self.human.height) , self.human.x:self.human.x+self.human.width]:
            self.human.set_object(vy=0)
            for y_i in range(self.human.y, self.human.y+self.human.height):
                if 255 in self.mask[min(y_i+self.human.height, self.height-self.human.height), self.human.x:self.human.x+self.human.width]:
                    self.human.set_object(y=y_i)
                    break
        self.human.update_object(height_max=self.height, width_max=self.width, dvy=G)
        if self.monster.x == 0:
            self.monster.set_object(vx=7)
        elif self.monster.x == self.width-self.monster.width:
            self.monster.set_object(vx=-7)
        self.monster.update_object(height_max=self.height, width_max=self.width)
        self.display_all_object()
        self.__display_maskedgh()
        if not self.is_clear and not self.is_gameover:
            for coin in self.coins:
                if coin.is_enable and self.judge_touch_object(self.human, coin):
                    coin.is_enable = False
                    self.score += 1
        if self.judge_touch_object(self.human, self.flag):
            self.is_clear = True
        if self.is_clear and not self.is_gameover:
            cv2.putText(self.frame, "CLEAR!!!", (20, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), thickness=4)
            cv2.putText(self.frame, "Press \'r\' to restart.", (50, self.height//4*3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3)
        if self.judge_touch_object(self.human, self.monster):
            self.is_gameover = True
        if self.is_gameover and not self.is_clear:
            cv2.putText(self.frame, "GAME OVER...", (20, self.height//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), thickness=4)
            cv2.putText(self.frame, "Press \'r\' to restart.", (50, self.height//4*3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=3)

        cv2.putText(self.frame, USAGE, (10, self.height//20), cv2.FONT_HERSHEY_COMPLEX , 0.5, (0, 0, 0), thickness=2)
        cv2.putText(self.frame, f"Score: {self.score}", (10, self.height//20+40), cv2.FONT_HERSHEY_COMPLEX , 1, (255, 0, 255), thickness=2)
        cv2.imshow('mask', self.mask)
        cv2.imshow('frame', self.frame)
        return True

    def __call__(self) -> None:
        result = True
        while result is True:
            result = self.run()
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    game = Game(debug=False)
    game()


if __name__ == "__main__":
    main()