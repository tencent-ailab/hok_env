"""
    created by timjxchen, 2021/7/27
    a simple video utils for visualize the game
"""

#
# require opencv
# just for test
import cv2
import numpy as np


class GameRender:
    WIDTH = 80000
    HEIGHT = 80000
    FPS = 15
    SCALE = 80 * 2

    CAMP_COLOR = [(50, 255, 50), (50, 50, 255), (255, 50, 50)]
    UNIT_SIZE = [8, 1, None]
    UNIT_THICK = [1, 2, 1]

    ORGAN_RANGE = {21: 1200, 24: 2560}

    def __init__(self, frame_frequency=1, dump_path="./"):
        self.video_shape = (self.WIDTH // self.SCALE, self.HEIGHT // self.SCALE)
        self.frame_frequency = frame_frequency
        self.dump_path = dump_path
        self.max_frames = 1000
        self.image_buffer = [
            np.zeros((self.WIDTH // self.SCALE, self.HEIGHT // self.SCALE, 3), np.uint8)
            for _ in range(self.max_frames)
        ]
        self.cur_b_index = 0
        self.cur_game_id = "0"
        self.video_writer = None

        self.camp_list = []
        cv2.circle(self.image_buffer[0], (10, 10), 2, (255, 255, 255), 1)

    def reset(self, game_id):
        self.cur_b_index = 0
        self.camp_list = []
        self.cur_game_id = game_id

    def draw_frame(self, req_pbs_in, frame_no):
        if frame_no % self.frame_frequency != 0:
            return
        req_pbs = []
        for r in req_pbs_in:
            if r is not None:
                req_pbs.append(r)

        self.image_buffer[self.cur_b_index][:] = 100
        player_ids = []
        camps = []
        full_info = len(req_pbs) == 2

        for id, req_pb in enumerate(req_pbs):
            player_id = req_pb.command_info_list[id].player_id
            camp = 0
            for state in req_pb.hero_list:
                if state.runtime_id == player_id:
                    camp = state.camp
                    break
            player_ids.append(player_id)
            camps.append(camp)

            for state in req_pb.soldier_list + req_pb.monster_list:
                if (not full_info) or state.camp == camp:
                    self._draw_unit(state, 1)

            for state in req_pb.organ_list:
                if (not full_info) or state.camp == camp:
                    self._draw_unit(state, 2)

            for state in req_pb.bullet_list:
                if (not full_info) or state.camp == camp:
                    self._draw_bullet(state)

            # for state in req_pb.frame_state.cakes:
            #     if (not full_info) or state.camp == camp:
            #         self._draw_unit(state)

        for id, req_pb in enumerate(req_pbs):
            player_id = player_ids[id]
            camp = camps[id]
            for state in req_pb.hero_list:
                if (not full_info) or state.camp == camp:
                    self._draw_unit(state, 0)

        # draw info as text
        self._draw_info(req_pbs, frame_no)

        # next frame
        self.cur_b_index += 1
        if self.cur_b_index >= self.max_frames:
            self.cur_b_index -= 1
            self._create_video_file()
            self._dump_cur_buffer()
            # print("warning: out of buffer")

    def dump_one_frame(self, idx=None):
        # (*'DVIX') or (*'X264')
        # if not working, pls install ffmepg: sudo apt-get install ffmepg
        if idx is None:
            idx = self.cur_b_index // 2
        file_name = "replay_{}.png".format(self.cur_game_id)
        frame = self.image_buffer[idx]
        cv2.imwrite(
            self.dump_path + file_name, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        )
        print(
            "dump frame {}/ {} at {}".format(
                idx, self.cur_b_index, self.dump_path + file_name
            )
        )

    def dump_one_round(self):
        self._create_video_file()
        # generate preview
        self.dump_one_frame()
        self._dump_cur_buffer()
        # generate preview
        self.video_writer.release()
        self.video_writer = None

    def init_camp(self, camp_list):
        self.camp_list = camp_list.copy()

    def rescale_color(self, color, factor):
        color = np.array(color, np.uint8)
        color = color * factor
        color = color.astype(np.uint8).tolist()
        return color

    def _dump_cur_buffer(self):
        v_writer = self.video_writer
        for i in range(self.cur_b_index + 1):
            frame = self.image_buffer[i]
            # frame = cv2.imread(img_root + str(i) + '.jpg')
            v_writer.write(frame)

        self.cur_b_index = 0

    def _create_video_file(self):
        if self.video_writer is not None:
            return
        # dump
        fps = 1 / self.frame_frequency * self.FPS
        # (*'DVIX') or (*'X264')
        # if not working, pls install ffmepg: sudo apt-get install ffmepg
        file_name = "replay_{}.avi".format(self.cur_game_id)
        # print("dump video {} at {}".format(self.cur_b_index, self.dump_path + file_name))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.video_writer = cv2.VideoWriter(
            self.dump_path + file_name, fourcc, fps, self.video_shape
        )

    def _draw_info(self, req_pbs, frame_no):
        # draw frame_no
        str = "frame: {}".format(frame_no)
        self._draw_text(str, (10, 50))

        # draw hero_location & hp

    def _draw_bullet(self, bullet):
        location = bullet.location
        pos = np.array([location.x, location.z])
        # if bullet.
        # forward = bullet.use_dir
        # forward = np.array([forward.x, forward.z])
        camp = bullet.camp
        color = self.CAMP_COLOR[camp]
        # draw hero
        # if bullet.source_actor:

        self._draw_circle(pos, color, 4, 1)
        # self._draw_line(pos, pos + forward, color, 1, 1)

    def _draw_unit(self, actor_state, actor_type):
        location = actor_state.location
        pos = np.array([location.x, location.z])
        camp = actor_state.camp
        if actor_state.hp == 0:
            return

        if actor_state.max_hp > 0:
            color_factor = actor_state.hp * 1.0 / actor_state.max_hp * (0.9) + 0.1
        else:
            color_factor = 1

        color = self.rescale_color(self.CAMP_COLOR[camp], color_factor)

        if actor_type == 0:
            # draw hero
            forward = actor_state.forward
            forward = np.array([forward.x, forward.z])
            self._draw_circle(pos, color, self.UNIT_SIZE[0], self.UNIT_THICK[0])
            self._draw_line(pos, pos + forward, color, 1, 1)
        elif actor_type == 1:
            # draw monster/soldier
            self._draw_circle(pos, color, self.UNIT_SIZE[0], self.UNIT_THICK[0])
            self._draw_circle(pos, color, self.UNIT_SIZE[1], self.UNIT_THICK[1])
        elif actor_type == 2:
            # draw organ
            self._draw_circle(pos, color, self.UNIT_SIZE[0], self.UNIT_THICK[0])
            if self.ORGAN_RANGE.get(actor_state.type) is not None:
                size = self.ORGAN_RANGE[actor_state.type] // self.SCALE
                self._draw_circle(pos, color, size, self.UNIT_THICK[2])
        else:
            self._draw_circle(pos, color)

    def _transpose_axis(self, p):
        x, z = p[0], p[1]
        pixel_x = (x + 40000) // self.SCALE
        pixel_z = (40000 - z) // self.SCALE
        return (pixel_x, pixel_z)

    def _draw_element(self, e_type, pos_list, color=None, size=2, thickness=0):
        if color is None:
            color = (255, 255, 255)
        if e_type == "circle":
            p = self._transpose_axis(pos_list)
            cv2.circle(self.image_buffer[self.cur_b_index], p, size, color, thickness)
        elif e_type == "line":
            fpos = self._transpose_axis(pos_list[0])
            tpos = self._transpose_axis(pos_list[1])
            cv2.line(
                self.image_buffer[self.cur_b_index], fpos, tpos, color, size, thickness
            )

    def _draw_circle(self, pos, color=None, size=2, thickness=0):
        if color is None:
            color = (255, 255, 255)
        p = self._transpose_axis(pos)

        # print('circle', p)
        cv2.circle(self.image_buffer[self.cur_b_index], p, size, color, thickness)

    def _draw_line(self, fpos, tpos, color=None, size=2, thickness=0):
        if color is None:
            color = (255, 255, 255)
        fpos = self._transpose_axis(fpos)
        tpos = self._transpose_axis(tpos)

        # print('line', fpos, tpos)
        cv2.line(
            self.image_buffer[self.cur_b_index], fpos, tpos, color, size, thickness
        )

    def _draw_text(self, str, pos, color=None, size=0.5, thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print('text', pos)
        cv2.putText(
            self.image_buffer[self.cur_b_index],
            str,
            pos,
            font,
            size,
            color,
            thickness,
            bottomLeftOrigin=False,
        )
