import math


class LRSchedule:
    """Learning rate schedule with warmup, main phase, and optional cooldown."""
    
    def __init__(self, base_lr: float, max_steps: int, warmup_steps: int, 
                 cooldown_steps: int, final_lr: float, schedule_type: str):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = max(0, warmup_steps)
        self.cooldown_steps = max(0, cooldown_steps)
        self.final_lr = final_lr
        self.schedule_type = schedule_type.lower()
        
        # Main schedule covers steps between warmup and cooldown
        self.main_steps = max_steps - self.warmup_steps - self.cooldown_steps
        if self.main_steps < 0:
            self.main_steps = 0

    def get_lr(self, step: int) -> float:
        # Warmup phase: linear ramp from 0 to base_lr
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        
        # Cooldown phase: 1-sqrt schedule to zero
        if self.cooldown_steps > 0 and step >= (self.max_steps - self.cooldown_steps):
            cooldown_step = step - (self.max_steps - self.cooldown_steps)
            progress = cooldown_step / self.cooldown_steps
            return self.final_lr * (1.0 - math.sqrt(progress))
        
        # Main phase
        if self.main_steps <= 0:
            return self.base_lr
            
        main_step = step - self.warmup_steps
        progress = min(1.0, main_step / self.main_steps)
        
        if self.schedule_type == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay
        elif self.schedule_type == "linear":
            return self.base_lr + (self.final_lr - self.base_lr) * progress
        elif self.schedule_type == "wsd":
            # Warmup-Stable-Decay: stable LR in main phase
            return self.base_lr
        else:
            # Default to cosine
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay
