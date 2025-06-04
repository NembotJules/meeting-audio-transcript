from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptTemplate:
    name: str
    template: str
    description: str

class PromptManager:
    def __init__(self):
        """Initialize with default prompts."""
        self.prompts = {
            "default": PromptTemplate(
                name="default",
                template="""
                Vous êtes un assistant IA chargé d'extraire des informations clés d'un transcript de réunion en français.
                
                **Contexte des réunions précédentes** :
                {context}
                
                **Transcript de la réunion actuelle** :
                {transcript}
                
                **Instructions** :
                Extraire les informations suivantes et les structurer dans un objet JSON avec les clés en anglais.
                Si une information fait référence à une réunion précédente, utilisez le contexte fourni pour la clarifier.
                
                Informations requises :
                1. Liste de présence (presence_list)
                2. Ordre du jour (agenda_items)
                3. Président (president)
                4. Rapporteur (rapporteur)
                5. Heures de début et fin (start_time, end_time)
                6. Revue des activités (activities_review)
                7. Résolutions (resolutions_summary)
                8. Sanctions (sanctions_summary)
                9. Solde du compte (balance_amount, balance_date)
                
                Pour chaque activité ou résolution qui fait référence à "la semaine dernière" ou "la dernière fois",
                utilisez le contexte pour fournir les détails complets.
                """,
                description="Default prompt with basic instructions"
            ),
            
            "detailed": PromptTemplate(
                name="detailed",
                template="""
                Vous êtes un expert en analyse de réunions bancaires, spécialisé dans l'extraction d'informations précises.
                
                **Contexte Historique** :
                {context}
                
                **Réunion Actuelle** :
                Date: {date}
                Titre: {title}
                Transcript:
                {transcript}
                
                **Instructions Détaillées** :
                
                1. PRÉSENCE (presence_list):
                   - Identifiez explicitement les présents ET les absents
                   - Si quelqu'un parle mais n'est pas dans la liste des présents, ajoutez-le
                   
                2. ORDRE DU JOUR (agenda_items):
                   - Numérotez en chiffres romains (I-, II-, etc.)
                   - Conservez la hiérarchie des points
                   
                3. ACTIVITÉS (activities_review):
                   Pour chaque membre:
                   - Dossiers en cours
                   - Actions réalisées
                   - Résultats obtenus
                   - Si "continuation" mentionnée, référencez le contexte
                   
                4. RÉSOLUTIONS (resolutions_summary):
                   Pour chaque résolution:
                   - Date précise
                   - Responsable
                   - Échéance
                   - Statut actuel
                   - Lien avec résolutions précédentes si mentionné
                   
                5. SANCTIONS (sanctions_summary):
                   Pour chaque sanction:
                   - Nom complet
                   - Motif détaillé
                   - Montant exact en FCFA
                   - Date d'application
                   - Statut de paiement
                   
                6. INFORMATIONS FINANCIÈRES:
                   - Solde exact (balance_amount)
                   - Date du solde (balance_date)
                   - Variations depuis la dernière réunion
                
                **Format de Sortie** :
                Retournez un JSON structuré avec toutes ces informations.
                Pour chaque référence à "la dernière fois" ou "la semaine passée",
                complétez avec les informations du contexte historique.
                """,
                description="Detailed prompt with specific instructions for banking context"
            )
        }
    
    def get_prompt(self, name: str = "default") -> PromptTemplate:
        """Get a prompt template by name."""
        return self.prompts.get(name, self.prompts["default"])
    
    def add_prompt(self, name: str, template: str, description: str) -> None:
        """Add a new prompt template."""
        self.prompts[name] = PromptTemplate(name=name, template=template, description=description)
    
    def format_prompt(self, 
                     name: str = "default", 
                     context: str = "", 
                     transcript: str = "",
                     date: str = "",
                     title: str = "") -> str:
        """Format a prompt template with the given context and transcript."""
        prompt = self.get_prompt(name)
        return prompt.template.format(
            context=context,
            transcript=transcript,
            date=date,
            title=title
        )
    
    def evaluate_prompt(self, 
                       generated_output: Dict,
                       ground_truth: Dict) -> Dict[str, float]:
        """
        Evaluate the quality of information extraction by comparing with ground truth.
        Returns metrics for different aspects of the extraction.
        """
        metrics = {}
        
        # Presence accuracy
        if "presence_list" in ground_truth:
            presence_match = self._calculate_presence_accuracy(
                generated_output.get("presence_list", ""),
                ground_truth["presence_list"]
            )
            metrics["presence_accuracy"] = presence_match
        
        # Activities completeness
        if "activities_review" in ground_truth:
            activities_score = self._calculate_activities_score(
                generated_output.get("activities_review", []),
                ground_truth["activities_review"]
            )
            metrics["activities_score"] = activities_score
        
        # Resolutions accuracy
        if "resolutions_summary" in ground_truth:
            resolutions_score = self._calculate_resolutions_score(
                generated_output.get("resolutions_summary", []),
                ground_truth["resolutions_summary"]
            )
            metrics["resolutions_score"] = resolutions_score
        
        # Calculate overall score
        metrics["overall_score"] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _calculate_presence_accuracy(self, generated: str, truth: str) -> float:
        """Calculate accuracy of presence list extraction."""
        # Simple string matching for now - could be enhanced with more sophisticated comparison
        generated_names = set(n.strip() for n in generated.replace("Présents:", "").replace("Absents:", "").split(","))
        truth_names = set(n.strip() for n in truth.replace("Présents:", "").replace("Absents:", "").split(","))
        
        if not truth_names:
            return 1.0 if not generated_names else 0.0
            
        intersection = len(generated_names.intersection(truth_names))
        union = len(generated_names.union(truth_names))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_activities_score(self, generated: List[Dict], truth: List[Dict]) -> float:
        """Calculate completeness and accuracy of activities extraction."""
        if not truth:
            return 1.0 if not generated else 0.0
            
        total_score = 0
        for truth_activity in truth:
            best_match_score = 0
            for gen_activity in generated:
                match_score = sum(
                    1 for key in truth_activity
                    if key in gen_activity and truth_activity[key] == gen_activity[key]
                ) / len(truth_activity)
                best_match_score = max(best_match_score, match_score)
            total_score += best_match_score
            
        return total_score / len(truth)
    
    def _calculate_resolutions_score(self, generated: List[Dict], truth: List[Dict]) -> float:
        """Calculate accuracy of resolutions extraction."""
        if not truth:
            return 1.0 if not generated else 0.0
            
        total_score = 0
        for truth_resolution in truth:
            best_match_score = 0
            for gen_resolution in generated:
                match_score = sum(
                    1 for key in truth_resolution
                    if key in gen_resolution and truth_resolution[key] == gen_resolution[key]
                ) / len(truth_resolution)
                best_match_score = max(best_match_score, match_score)
            total_score += best_match_score
            
        return total_score / len(truth) 