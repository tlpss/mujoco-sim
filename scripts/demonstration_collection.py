
import enum
import time 
import numpy as np 
class DatasetRecorder:
    def start_episode(self):
        raise NotImplementedError

    def record(self, obs, action, reward, done, info):
        raise NotImplementedError

    def save_episode(self):
        raise NotImplementedError

class DummyDatasetRecorder(DatasetRecorder):
    def start_episode(self):
        print("starting dataset episode recording")

    def record(self, obs, action, reward, done, info):
        print("recording step")

    def save_episode(self):
        print("saving dataset episode")



        
    
class UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Demonstration Collection")
        self.clock = pygame.time.Clock()

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "recording"
                elif event.key == pygame.K_d:
                    return "discard"
                elif event.key == pygame.K_f:
                    return "finish"
                elif event.key == pygame.K_q:
                    return "quit"
        return None

    def update_display_color(self, color):
        self.screen.fill(color)
        pygame.display.flip()

    def add_render_to_screen(self, img: np.ndarray):
        assert img.shape[2] == 3, "Image should be in RGB format"
        print(img.shape)
        img = img * 255

        img = np.ones((100,100,3), dtype=np.uint8) * 255
        img = pygame.surfarray.make_surface(img)
        self.screen.blit(img, (50,50))
        pygame.display.flip()

class StateMachine:
    STATES = enum.Enum("states", "waiting start recording discard finish quit")

    def __init__(self):
        self.state = self.STATES.waiting
        self.ui = UI()

    def update_state_from_keyboard(self):
        event = self.ui.process_events()
        if event:
            self.state = self.STATES[event]
        #print(f"State: {self.state}")

        if self.state is self.STATES.recording:
            self.ui.update_display_color(( 0, 0,255))
        else:
            self.ui.update_display_color((0, 0, 0))


    

def collect_demonstrations(agent_callable, env, dataset_recorder):
    # start the UI
    state_machine = StateMachine()
    while state_machine.state is not state_machine.STATES.quit:
        while state_machine.state is state_machine.STATES.waiting:
            time.sleep(1)
            state_machine.update_state_from_keyboard()
        

        env.reset()
        dataset_recorder.start_episode()
        done = False

        while not done and state_machine.state is state_machine.STATES.recording:
            action = agent_callable(env)
            obs, reward, done, info = env.step(action)
            dataset_recorder.record(obs, action, reward, done, info)
            img = env.render()
            # add img to the screen
            state_machine.update_state_from_keyboard()
            state_machine.ui.add_render_to_screen(img)


        print("Episode is done, decide what to do next")
        while done and state_machine.state is state_machine.STATES.recording:
            state_machine.update_state_from_keyboard()

            
        if state_machine.state is state_machine.STATES.discard:
            print("not saving the episode")

        if state_machine.state is state_machine.STATES.finish:
            dataset_recorder.save_episode()

        state_machine.state = state_machine.STATES.waiting





        
if __name__ =="__main__":
    from mujoco_sim.environments.tasks.point_reach import PointMassReachTask, PointReachConfig, create_demonstation_policy
    from mujoco_sim.environments.dmc2gym import DMCWrapper
    from dm_control.composer import Environment as DMEnvironment
    import pygame

    task = PointMassReachTask(PointReachConfig())
    dmc_env = DMEnvironment(task)
    env = DMCWrapper(dmc_env)

    action_callable = create_demonstation_policy(dmc_env)

    dataset_recorder = DummyDatasetRecorder()
    collect_demonstrations(action_callable, env, dataset_recorder)

