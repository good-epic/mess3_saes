# KDE - null:
PYTHONPATH=. python -u validation/latent_spatial_simplex.py --simplex_samples_dir outputs/null_simplex_samples --vertex_acts_dir outputs/selected_null_clusters --output_dir outputs/validation/latent_spatial_null --clusters 512_138,512_345,768_310 --heatmap_grid_size 100

# KDE - real:
PYTHONPATH=. python -u validation/latent_spatial_simplex.py --simplex_samples_dir outputs/simplex_samples --vertex_acts_dir outputs/selected_clusters_broad_2 --output_dir outputs/validation/latent_spatial --clusters 512_17,512_22,512_67,512_181,512_229,512_261,512_471,512_504,768_140,768_210,768_306,768_581,768_596 --heatmap_grid_size 100

# Prepare null samples:
PYTHONPATH=. python -u interpretation/prepare_vertex_samples.py --manifest outputs/selected_null_clusters/manifest.json --output_dir outputs/interpretations/prepared_samples_null --max_samples_per_vertex 200 --min_samples_per_vertex 15

# Interpret null clusters:
PYTHONPATH=. python -u interpretation/interpret_clusters_dual_path.py --prepared_samples_dir outputs/interpretations/prepared_samples_null --path_a_template interpretation/prompt_templates/detailed_all_vertices.txt --path_b_vertex_template interpretation/prompt_templates/detailed_one_vertex.txt --path_b_synthesis_template interpretation/prompt_templates/detailed_synthesis.txt --output_dir outputs/interpretations/sonnet_null --analysis_mode both --model sonnet --samples_per_vertex 30 --num_iterations 20 --clusters_to_process 512_138,512_345,768_310 --seed 42
