import logging
import numpy as np

logger = logging.getLogger("traffic-monitor.direction")


class DirectionChecker:
    """
    Detecta veículos na direção contrária.
    
    Define uma direção "correta" de tráfego e verifica se
    o vetor de movimento do veículo está no sentido oposto.
    """
    
    def __init__(self, config: dict):
        self.expected = config.get("expected", "left_to_right")
        self.tolerance = config.get("tolerance", 2.0)
        self.min_positions = 5  # Mínimo de posições pra determinar direção
        
        logger.info(f"Direção esperada: {self.expected}")
    
    def check(self, track_history: list) -> dict:
        """
        Verifica a direção do veículo.
        
        Args:
            track_history: Lista de tuplas (x, y) em pixels
        
        Returns:
            {
                "direction": "left_to_right" | "right_to_left" | "unknown",
                "expected": str,
                "is_wrong_way": bool,
                "confidence": float (0-1),
            }
        """
        result = {
            "direction": "unknown",
            "expected": self.expected,
            "is_wrong_way": False,
            "confidence": 0.0,
        }
        
        if len(track_history) < self.min_positions:
            return result
        
        # Calcular deslocamento total no eixo X
        # Usar regressão linear para ser mais robusto contra ruído
        recent = track_history[-min(20, len(track_history)):]
        x_coords = np.array([p[0] for p in recent])
        y_coords = np.array([p[1] for p in recent])
        
        # Deslocamento médio entre frames consecutivos
        dx_values = np.diff(x_coords)
        avg_dx = np.mean(dx_values)
        
        # Desvio padrão para confiança
        if len(dx_values) > 1:
            std_dx = np.std(dx_values)
            # Se o desvio é alto, o movimento é inconsistente (curvas, etc.)
            # Confiança é maior quando o movimento é consistente
            result["confidence"] = min(1.0, abs(avg_dx) / (std_dx + 1e-6))
            result["confidence"] = min(result["confidence"], 1.0)
        else:
            result["confidence"] = 0.5
        
        # Ignorar se o movimento é muito pequeno (veículo parado ou muito lento)
        if abs(avg_dx) < self.tolerance:
            return result
        
        # Determinar direção
        if avg_dx > 0:
            result["direction"] = "left_to_right"
        else:
            result["direction"] = "right_to_left"
        
        # Verificar se é direção contrária
        if result["direction"] != "unknown" and result["direction"] != self.expected:
            result["is_wrong_way"] = True
            # Apenas alertar se a confiança é razoável
            if result["confidence"] < 0.3:
                result["is_wrong_way"] = False  # Provável falso positivo
        
        return result
