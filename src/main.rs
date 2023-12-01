use std::f32::consts::PI;
use std::f32::consts::TAU;
use std::future;

use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::color::Color;
use bevy::render::mesh::Indices;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::tasks;
use bevy::tasks::AsyncComputeTaskPool;
use bevy::tasks::Task;
use bevy::tasks::TaskPool;
use bevy::utils::hashbrown::HashMap;
use bevy::utils::hashbrown::HashSet;
use bevy::window::PrimaryWindow;
use bitflags::bitflags;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Block {
    Void,
    Air,
    Stone,
}

pub trait Structure {
    fn new() -> Self
    where
        Self: Sized;

    fn get_block(&self, position: UVec3) -> Block;
    fn set_block(&mut self, position: UVec3, block: Block);
    fn get_cull(&self, position: UVec3) -> Direction;
    fn set_cull(&mut self, position: UVec3, direction: Direction);

    fn size(&self) -> UVec3;

    fn linearize(&self, position: UVec3) -> usize {
        let UVec3 { y: sy, x: sx, .. } = self.size();
        let UVec3 { x, y, z } = position;
        ((z * sy + y) * sx + x) as usize
    }

    fn delinearize(&self, index: usize) -> UVec3 {
        let UVec3 {
            x: sx,
            y: sy,
            z: sz,
        } = self.size();
        let mut idx = index as u32;
        let z = idx / (sx * sy);
        idx -= (z * sx * sy);
        let y = idx / sx;
        let x = idx % sx;
        UVec3 { x, y, z }
    }

    fn count(&self) -> usize {
        let UVec3 { x, y, z } = self.size();
        (x * y * z) as usize
    }
}

#[derive(Clone, Copy)]
pub struct ChunkBlockInfo {
    block: Block,
    cull_faces: Direction,
}

pub struct Chunk {
    data: Vec<ChunkBlockInfo>,
}

impl Structure for Chunk {
    fn new() -> Self
    where
        Self: Sized,
    {
        Chunk {
            data: vec![
                ChunkBlockInfo {
                    block: Block::Void,
                    cull_faces: Direction::empty()
                };
                64 * 64 * 64
            ],
        }
    }

    fn get_block(&self, position: UVec3) -> Block {
        let index = self.linearize(position);
        self.data[index].block
    }

    fn set_block(&mut self, position: UVec3, block: Block) {
        let index = self.linearize(position);
        self.data[index].block = block
    }

    fn get_cull(&self, position: UVec3) -> Direction {
        let index = self.linearize(position);
        self.data[index].cull_faces
    }

    fn set_cull(&mut self, position: UVec3, direction: Direction) {
        let index = self.linearize(position);
        self.data[index].cull_faces = direction;
    }

    fn size(&self) -> UVec3 {
        UVec3::new(64, 64, 64)
    }
}

bitflags! {
    #[derive(PartialEq, Eq, Clone, Copy)]
    pub struct Direction: usize {
        const LEFT =    0b00000001;
        const RIGHT =   0b00000010;
        const DOWN =    0b00000100;
        const UP =      0b00001000;
        const BACK =    0b00010000;
        const FOWARD =  0b00100000;
        const ALL =  0b00111111;
    }
}

fn cube_mesh_parts(
    position: Vec3,
    directions: Direction,
    vertices: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let cube_vertices = [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
    ];

    let cube_normals = [
        [
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    ];

    let cube_indices = [
        [4, 5, 7, 5, 6, 7],
        [0, 3, 1, 1, 3, 2],
        [12, 13, 15, 13, 14, 15],
        [8, 11, 9, 9, 11, 10],
        [20, 21, 23, 21, 22, 23],
        [16, 19, 17, 17, 19, 18],
    ];

    for current_direction in (0..6)
        .map(|x| 1 << x)
        .map(Direction::from_bits)
        .map(Option::unwrap)
    {
        if current_direction & directions == Direction::empty() {
            continue;
        }
        let index = current_direction.bits().trailing_zeros() as usize;

        let count = vertices.len();

        vertices.extend(
            cube_vertices[index]
                .iter()
                .map(|unit| (Vec3::from_array(*unit) + position).to_array()),
        );
        normals.extend(cube_normals[index].iter());
        indices.extend(cube_indices[index].iter().map(|i| (count + i % 4) as u32))
    }
}

#[rustfmt::skip]
fn create_structure_mesh(structure: &dyn Structure) -> Mesh {
    let mut vertices = vec![];
    let mut normals = vec![];
    let mut indices = vec![];


    for index in 0..structure.count() {
        let position = structure.delinearize(index);
        if !matches!(structure.get_block(position), Block::Air) {
            cube_mesh_parts(position.as_vec3(), structure.get_cull(position), &mut vertices, &mut normals, &mut indices);
        }
    }
dbg!(vertices.len());
    Mesh::new(PrimitiveTopology::TriangleList)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
            vertices
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals
    )
    .with_indices(Some(Indices::U32(indices)))
}

fn camera(
    mut query1: Query<(&Parent, &Camera)>,
    mut query2: Query<(&mut Transform)>, keys: Res<Input<KeyCode>>, time: Res<Time>, 
) {
    let (camera_parent, _) = query1.single_mut();
    let (mut parent_transform) = query2.get_mut(camera_parent.get()).unwrap();

    let rotation = keys.pressed(KeyCode::E) as i32 - keys.pressed(KeyCode::Q) as i32;

    parent_transform.rotate_y(time.delta_seconds() * 0.25 * TAU * rotation as f32);

    let speed = 50.4;
    let lateral_direction = IVec3 {
        x: keys.pressed(KeyCode::D) as i32 - keys.pressed(KeyCode::A) as i32,
        y: 0,
        z: keys.pressed(KeyCode::S) as i32 - keys.pressed(KeyCode::W) as i32,
    };
    let vertical_direction = IVec3 {
        x: 0,
        y: keys.pressed(KeyCode::Space) as i32 - keys.pressed(KeyCode::ShiftLeft) as i32,
        z: 0,
    };
    let rotation = Quat::from_axis_angle(Vec3::Y, parent_transform.rotation.to_euler(EulerRot::YXZ).0);
    let movement = speed
        * time.delta_seconds()
        * (rotation * lateral_direction.as_vec3() + vertical_direction.as_vec3())
            .normalize_or_zero();
    parent_transform.translation += movement;
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    dbg!("yo");
    let mut camera_transform = Transform::from_xyz(0.0, 1000.0, 1000.0);
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
    let camera = commands.spawn(Camera3dBundle {
        projection: Projection::Perspective(PerspectiveProjection { fov: PI / 24.0, aspect_ratio: 16.0 / 9.0, near: 0.1, far: 10000.0 }),
        transform: camera_transform,
        ..default()
    }).id();
    commands.spawn((GlobalTransform::default(), Transform::from_xyz(0.0, 0.0, 0.0))).push_children(&[camera]);
    let mut light_transform = Transform::from_xyz(1000.0, 10000.0, 1000.0);
    light_transform.look_at(Vec3::ZERO, Vec3::Y);
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::Rgba {
                red: 1.0,
                green: 0.996,
                blue: 0.976,
                alpha: 1.0,
            },
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: light_transform,
        ..default()
    });
}


#[derive(Resource)]
pub struct World {
    view: usize,
    origin: IVec3,
    loaded: HashSet<IVec3>,
    mapping: HashMap<IVec3, Entity>,
    chunk_futures: Vec<Task<(IVec3, Chunk)>>,
}

fn spawn(
    mut world: ResMut<World>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut not_done_futures = vec![];
    let mut future_drain_iter = world.chunk_futures.drain(..);
    let max = 1;
    let mut count = 0;
    while let Some(chunk_future) = future_drain_iter.next() {
        if count >= max {
            not_done_futures.push(chunk_future);
            break;
        }
        if !chunk_future.is_finished() {
            not_done_futures.push(chunk_future);
            continue;
        }
        dbg!("done");
        tasks::block_on(async {
            let (position, chunk) = chunk_future.await;

            let cube_mesh_handle: Handle<Mesh> = meshes.add(create_structure_mesh(&chunk));
            dbg!("spawned");
            commands.spawn(
                (PbrBundle {
                    mesh: cube_mesh_handle,
                    material: materials.add(StandardMaterial {
                        base_color: Color::Rgba {
                            red: 1.0,
                            green: 1.0,
                            blue: 1.0,
                            alpha: 1.0,
                        },
                        ..default()
                    }),
                    transform: Transform {
                        translation: position.as_vec3() * 64.0,
                        ..default()
                    },
                    ..default()
                }),
            );
        });
        count += 1;
    }
    not_done_futures.extend(future_drain_iter);
    world.chunk_futures = not_done_futures;
}

fn load(
    query1: Query<(&Camera, &Parent)>,
    query2: Query<(&Transform)>,
    mut world: ResMut<World>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (_, camera_parent) = query1.single();
    let camera_transform = query2.get(camera_parent.get()).unwrap();
    let position = camera_transform.translation.as_ivec3() / 64;
    dbg!(position);
    if position == world.origin {
        return;
    }

    world.origin = position;

    let mut needed = HashSet::new();

    let view = world.view as i32;
    for x in -view..=view {
        for y in 0..=2 {
            for z in -view..=view {
                needed.insert(world.origin + IVec3 { x, y, z });
            }
        }
    }

    let not_loaded = needed
        .difference(&world.loaded)
        .copied()
        .collect::<HashSet<_>>();

    for position in not_loaded {
        let chunk_future =
            AsyncComputeTaskPool::get_or_init(|| TaskPool::new()).spawn(async move {
                let mut chunk = gen_chunk(position);
                internal_cull_faces(&mut chunk);

                (position, chunk)
            });

        world.chunk_futures.push(chunk_future);
        world.loaded.insert(position);
    }
}

fn gen_chunk(position: IVec3) -> Chunk {
    let mut chunk = Chunk {
        data: vec![
            ChunkBlockInfo {
                block: Block::Void,
                cull_faces: Direction::empty()
            };
            64 * 64 * 64
        ],
    };
    let UVec3 {
        x: sx,
        y: sy,
        z: sz,
    } = chunk.size();
    let perlin = noise::Fbm::<noise::Perlin>::new(400);
    const NOISE_SCALE: i32 = 16;

    let mut noise_values = vec![];

    for z in 0..=64 / NOISE_SCALE {
        for y in 0..=64 / NOISE_SCALE {
            for x in 0..=64 / NOISE_SCALE {
                let nx = position.x * 64 + x * NOISE_SCALE;
                let ny = position.y * 64 + y * NOISE_SCALE;
                let nz = position.z * 64 + z * NOISE_SCALE;
                use noise::NoiseFn;
                let density =
                    perlin.get([nx as f64 * 0.0015, ny as f64 * 0.0015, nz as f64 * 0.0015]);
                noise_values.push(density);
            }
        }
    }

    fn lerp3d(
        xm_ym_zm: f64,
        xp_ym_zm: f64,
        xm_yp_zm: f64,
        xp_yp_zm: f64,
        xm_ym_zp: f64,
        xp_ym_zp: f64,
        xm_yp_zp: f64,
        xp_yp_zp: f64,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        (xm_ym_zm * (1.0 - x) * (1.0 - y) * (1.0 - z))
            + (xp_ym_zm * x * (1.0 - y) * (1.0 - z))
            + (xm_yp_zm * (1.0 - x) * y * (1.0 - z))
            + (xp_yp_zm * x * y * (1.0 - z))
            + (xm_ym_zp * (1.0 - x) * (1.0 - y) * z)
            + (xp_ym_zp * x * (1.0 - y) * z)
            + (xm_yp_zp * (1.0 - x) * y * z)
            + (xp_yp_zp * x * y * z)
    }

    let smx = sx as usize / NOISE_SCALE as usize + 1;
    let smy = sy as usize / NOISE_SCALE as usize + 1;

    for z in 0..64 {
        for x in 0..64 {
            for y in 0..64 {
                let ix = x as usize % NOISE_SCALE as usize;
                let iy = y as usize % NOISE_SCALE as usize;
                let iz = z as usize % NOISE_SCALE as usize;
                let ny = position.y * sy as i32 + y as i32;

                let mx0 = x as usize / NOISE_SCALE as usize;
                let my0 = y as usize / NOISE_SCALE as usize;
                let mz0 = z as usize / NOISE_SCALE as usize;

                let mx1 = mx0 + 1;
                let my1 = my0 + 1;
                let mz1 = mz0 + 1;

                let x0y0z0 = noise_values[(mz0 * smy + my0) * smx + mx0];
                let x1y0z0 = noise_values[(mz0 * smy + my0) * smx + mx1];
                let x0y1z0 = noise_values[(mz0 * smy + my1) * smx + mx0];
                let x0y0z1 = noise_values[(mz1 * smy + my0) * smx + mx0];
                let x1y1z0 = noise_values[(mz0 * smy + my1) * smx + mx1];
                let x0y1z1 = noise_values[(mz1 * smy + my1) * smx + mx0];
                let x1y0z1 = noise_values[(mz1 * smy + my0) * smx + mx1];
                let x1y1z1 = noise_values[(mz1 * smy + my1) * smx + mx1];

                let density = lerp3d(x0y0z0, x1y0z0,
                    x0y1z0, x1y1z0,
                    x0y0z1, x1y0z1,
                    x0y1z1, x1y1z1,
                    ix as f64 / NOISE_SCALE as f64, iy as f64 / NOISE_SCALE as f64, iz as f64 / NOISE_SCALE as f64);    

                let density_mod = (32isize - ny as isize) as f64 * 0.035;
                chunk.set_block(
                    UVec3 { x, y, z },
                    if density + density_mod > 0.0 {
                        Block::Stone
                    } else {
                        Block::Air
                    },
                );
            }
        }
    }
    chunk
}

fn internal_cull_faces(structure: &mut dyn Structure) {
    let UVec3 {
        x: sx,
        y: sy,
        z: sz,
    } = structure.size();

    for index in 0..structure.count() {
        let mut dir_iter = (0..6)
            .map(|x| 1 << x)
            .map(Direction::from_bits)
            .map(Option::unwrap);
        let mut directions = Direction::empty();

        let position = structure.delinearize(index);
        for d in 0..3 {
            for n in (-1..=1).step_by(2) {
                let current_direction = dir_iter.next().unwrap();
                let mut normal = IVec3::default();
                normal[d] = n;
                let normal = normal.as_uvec3();
                let neighbor = position + normal;
                if neighbor.x >= sx || neighbor.y >= sy || neighbor.z >= sz {
                    continue;
                }
                if neighbor.x < sx && neighbor.y < sy && neighbor.z < sz {
                    if matches!(structure.get_block(neighbor), Block::Air) {
                        directions |= current_direction;
                    }
                }
            }
        }

        structure.set_cull(position, directions);
    }
}

fn main() {
    dbg!("yo");
    let mut app = App::new();

    app.insert_resource(World {
        view: 8,
        origin: IVec3 {
            x: i32::MAX,
            y: 0,
            z: 0,
        },
        loaded: HashSet::new(),
        mapping: HashMap::new(),
        chunk_futures: Vec::new(),
    });

    app.add_plugins(DefaultPlugins);
    app.add_systems(Startup, setup)
        .add_systems(Update, load)
        .add_systems(Update, spawn)
        .add_systems(Update, camera);

    app.run();
}
