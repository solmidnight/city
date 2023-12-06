use std::f32::consts::PI;
use std::f32::consts::TAU;
use std::future;
use std::iter;
use std::mem;

use bevy::ecs::system::SystemState;
use bevy::input::mouse::MouseMotion;
use bevy::pbr::CascadeShadowConfig;
use bevy::pbr::CascadeShadowConfigBuilder;
use bevy::pbr::Cascades;
use bevy::pbr::DirectionalLightShadowMap;
use bevy::prelude::*;
use bevy::render::color::Color;
use bevy::render::mesh::Indices;
use bevy::render::mesh::VertexAttributeValues;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::tasks;
use bevy::tasks::AsyncComputeTaskPool;
use bevy::tasks::Task;
use bevy::tasks::TaskPool;
use bevy::tasks::TaskPoolBuilder;
use bevy::utils::hashbrown::HashMap;
use bevy::utils::hashbrown::HashSet;
use bevy::window::PrimaryWindow;
use bezier_nd::Bezier;
use bitflags::bitflags;
use voxels::Channel;

const CHUNK_AXIS: usize = 32;

#[repr(u64)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Block {
    Void,
    Air,
    Stone,
    Grass,
}

impl Block {
    fn color(self) -> Vec4 {
        match self {
            Block::Void => Vec4::new(1.0, 0.0, 1.0, 1.0),
            Block::Stone => Vec4::new(0.4, 0.4, 0.4, 1.0),
            Block::Grass => Vec4::new(0.0, 0.6, 0.09, 1.0),
            _ => Vec4::splat(0.0),
        }
    }
}

#[derive(Component)]
pub struct Dirty;

#[derive(Component)]
pub struct Active;

#[derive(Component)]
pub struct Chunk(IVec3);

#[derive(Component)]
pub struct Structure {
    size: UVec3,
    blocks: Channel,
    cull_faces: Channel,
    ao: Channel,
}

impl Structure {
    fn new(size: UVec3) -> Self {
        let UVec3 {
            x: sx,
            y: sy,
            z: sz,
        } = size;
        let mut blocks = Channel::default();
        blocks.extend(iter::repeat(Block::Void as u64).take((sx * sy * sz) as usize));
        let mut cull_faces = Channel::default();
        cull_faces.extend(iter::repeat(Direction::empty().bits()).take((sx * sy * sz) as usize));
        let mut ao = Channel::default();
        ao.extend(iter::repeat(0).take((sx * sy * sz) as usize));
        Structure {
            size,
            blocks,
            cull_faces,
            ao,
        }
    }

    fn get_block(&self, position: impl IntoIterator<Item = UVec3>) -> impl Iterator<Item = Block> {
        self.blocks
            .get(position.into_iter().map(|pos| self.linearize(pos) as u64))
            .into_iter()
            .map(|id| unsafe { mem::transmute(id) })
    }

    fn set_block(&mut self, data: impl IntoIterator<Item = (UVec3, Block)>) {
        let data = data
            .into_iter()
            .map(|(pos, block)| (self.linearize(pos) as u64, block as u64))
            .collect::<Vec<_>>();
        self.blocks.set(data);
    }

    fn get_cull(
        &self,
        position: impl IntoIterator<Item = UVec3>,
    ) -> impl Iterator<Item = Direction> {
        self.cull_faces
            .get(position.into_iter().map(|pos| self.linearize(pos) as u64))
            .into_iter()
            .map(|id| Direction::from_bits(id).unwrap())
    }

    fn set_cull(&mut self, data: impl IntoIterator<Item = (UVec3, Direction)>) {
        let data = data
            .into_iter()
            .map(|(pos, dir)| (self.linearize(pos) as u64, dir.bits()))
            .collect::<Vec<_>>();
        self.cull_faces.set(data);
    }

    fn get_ao(&self, position: impl IntoIterator<Item = UVec3>) -> impl Iterator<Item = [Vec4; 6]> {
        self.ao
            .get(position.into_iter().map(|pos| self.linearize(pos) as u64))
            .into_iter()
            .map(|data| {
                let mut ao = [Vec4::splat(0.0); 6];
                for x in 0..6 {
                    for y in 0..4 {
                        ao[x][y] = ((data >> ((x * 8) + y * 2)) & 3) as f32 / 3.0;
                    }
                }
                ao
            })
    }

    fn set_ao(&mut self, data: impl IntoIterator<Item = (UVec3, [Vec4; 6])>) {
        let data = data
            .into_iter()
            .map(|(pos, ao)| {
                let mut data = 0u64;
                for x in 0..6 {
                    for y in 0..4 {
                        data |= ((ao[x][y] * 3.0) as u64) << ((x * 8) + y * 2);
                    }
                }
                (self.linearize(pos) as u64, data)
            })
            .collect::<Vec<_>>();
        self.ao.set(data);
    }

    fn size(&self) -> UVec3 {
        self.size
    }

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

bitflags! {
    #[derive(PartialEq, Eq, Clone, Copy)]
    pub struct Direction: u64 {
        const LEFT =    0b00000001;
        const RIGHT =   0b00000010;
        const DOWN =    0b00000100;
        const UP =      0b00001000;
        const BACK =    0b00010000;
        const FORWARD =  0b00100000;
        const ALL =  0b00111111;
    }
}

impl Direction {
    fn opposite(self) -> Self {
        if self & Self::LEFT == Direction::empty() {
            Self::RIGHT
        } else if self & Self::RIGHT == Direction::empty() {
            Self::LEFT
        } else if self & Self::DOWN == Direction::empty() {
            Self::UP
        } else if self & Self::UP == Direction::empty() {
            Self::DOWN
        } else if self & Self::BACK == Direction::empty() {
            Self::FORWARD
        } else if self & Self::FORWARD == Direction::empty() {
            Self::UP
        } else {
            panic!("cannot have opposite of multiple directions");
        }
    }
}

fn cube_mesh_parts(
    position: Vec3,
    directions: Direction,
    color: Vec4,
    ao: [Vec4; 6],
    vertices: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
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
        let [a, b, c, d] = ao[index].to_array();
        colors.push((color * Vec4::new(c, c, c, 1.0)).to_array());
        colors.push((color * Vec4::new(b, b, b, 1.0)).to_array());
        colors.push((color * Vec4::new(a, a, a, 1.0)).to_array());
        colors.push((color * Vec4::new(d, d, d, 1.0)).to_array());
        normals.extend(cube_normals[index].iter());
        indices.extend(cube_indices[index].iter().map(|i| (count + i % 4) as u32))
    }
}

#[rustfmt::skip]
fn create_structure_mesh(structure: &Structure) -> Mesh {
    let mut vertices = vec![];
    let mut colors = vec![];
    let mut normals = vec![];
    let mut indices = vec![];

    let blocks = structure.get_block((0..structure.count()).map(|index| structure.delinearize(index))).collect::<Vec<_>>();
    let cull = structure.get_cull((0..structure.count()).map(|index| structure.delinearize(index))).collect::<Vec<_>>();
    let ao = structure.get_ao((0..structure.count()).map(|index| structure.delinearize(index))).collect::<Vec<_>>();
    for index in 0..structure.count() {
        let position = structure.delinearize(index);
        if !matches!(blocks[index], Block::Air) {
            cube_mesh_parts(position.as_vec3(), cull[index], blocks[index].color(), ao[index], &mut vertices, &mut colors, &mut normals, &mut indices);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
            vertices
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_COLOR,
            colors
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals
    )
    .with_indices(Some(Indices::U32(indices)))
}

fn camera(
    mut query1: Query<(&Parent, &Camera)>,
    mut query2: Query<(&mut Transform)>,
    keys: Res<Input<KeyCode>>,
    time: Res<Time>,
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
    let rotation =
        Quat::from_axis_angle(Vec3::Y, parent_transform.rotation.to_euler(EulerRot::YXZ).0);
    let movement =
        speed * time.delta_seconds() * (rotation * lateral_direction.as_vec3()).normalize_or_zero();
    parent_transform.translation += movement;
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut camera_transform = Transform::from_xyz(0.0, 1000.0, 1000.0);
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
    let camera = commands
        .spawn(Camera3dBundle {
            projection: Projection::Perspective(PerspectiveProjection {
                fov: PI / 36.0,
                aspect_ratio: 16.0 / 9.0,
                near: 0.1,
                far: 10000.0,
            }),
            transform: camera_transform,
            ..default()
        })
        .id();
    commands
        .spawn((
            GlobalTransform::default(),
            Transform::from_xyz(0.0, 0.0, 0.0),
        ))
        .push_children(&[camera]);
    let mut light_transform = Transform::from_xyz(1000.0, 1000.0, 1000.0);
    light_transform.look_at(Vec3::ZERO, Vec3::Y);
    let mut cascade_shadow_config_builder = CascadeShadowConfigBuilder::default();
    cascade_shadow_config_builder.first_cascade_far_bound = 1300.0;
    cascade_shadow_config_builder.minimum_distance = 1200.0;
    cascade_shadow_config_builder.maximum_distance = 2000.0;
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
        cascade_shadow_config: cascade_shadow_config_builder.build(),
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
    chunk_futures: Option<Vec<Task<(IVec3, Structure)>>>,
}

fn spawn(mut world: ResMut<World>, mut commands: Commands) {
    let (a, b) = world
        .chunk_futures
        .take()
        .unwrap()
        .into_iter()
        .partition(|future| future.is_finished());
    world.chunk_futures = Some(b);
    for chunk_future in a {
        let (position, chunk) = tasks::block_on(async { chunk_future.await });

        let entity = commands.spawn((chunk, Chunk(position), Dirty)).id();
        world.mapping.insert(position, entity);
    }
}

fn consolidate(bevy_world: &mut bevy::prelude::World) {
    let mut dirty_chunk_data = vec![];

    let mut system_state = SystemState::<(
        Res<World>,
        Query<(Entity, &Chunk), (With<Structure>, With<Dirty>)>,
    )>::new(bevy_world);
    let (world, query) = system_state.get(bevy_world);
    for (entity, Chunk(position)) in query.iter() {
        if !all_neighbors_present(&world.mapping, *position) {
            continue;
        }
        dirty_chunk_data.push((entity, Chunk(*position)));
    }
    drop(world);
    drop(query);
    drop(system_state);

    for (chunk_entity, Chunk(position)) in dirty_chunk_data {
        let mut blocks = vec![];
        let mut all = HashMap::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let neighbor_chunk_position = position + IVec3::new(x, y, z);
                    let &neighbor_entity = bevy_world
                        .resource_mut::<World>()
                        .mapping
                        .get(&neighbor_chunk_position)
                        .unwrap();

                    let neighbor_chunk = bevy_world.get::<Structure>(neighbor_entity).unwrap();

                    let range = (0..neighbor_chunk.count()).map(|i| neighbor_chunk.delinearize(i));
                    all.insert(
                        neighbor_chunk_position,
                        neighbor_chunk.get_block(range).collect::<Vec<_>>(),
                    );
                }
            }
        }
        for d in 0..3 {
            for n in -1..=1 {
                let mut neighbor = IVec3::ZERO;
                if n == -1 {
                    neighbor[d] = -1
                } else {
                    neighbor[d] = CHUNK_AXIS as i32;
                }
                for u in -1..=CHUNK_AXIS as i32 {
                    for v in -1..=CHUNK_AXIS as i32 {
                        neighbor[(d + 1) % 3] = u;
                        neighbor[(d + 2) % 3] = v;
                        let neighbor_chunk_position = position
                            + IVec3::new(
                                neighbor.x.div_euclid(CHUNK_AXIS as i32),
                                neighbor.y.div_euclid(CHUNK_AXIS as i32),
                                neighbor.z.div_euclid(CHUNK_AXIS as i32),
                            );
                        let &neighbor_entity = bevy_world
                            .resource_mut::<World>()
                            .mapping
                            .get(&neighbor_chunk_position)
                            .unwrap();
                        let local = IVec3::new(
                            neighbor.x.rem_euclid(CHUNK_AXIS as i32),
                            neighbor.y.rem_euclid(CHUNK_AXIS as i32),
                            neighbor.z.rem_euclid(CHUNK_AXIS as i32),
                        );
                        let neighbor_chunk = bevy_world.get::<Structure>(neighbor_entity).unwrap();

                        blocks.push((
                            (neighbor + 1).as_uvec3(),
                            all[&neighbor_chunk_position]
                                [neighbor_chunk.linearize(local.as_uvec3())],
                        ));

                        let true_position = IVec3::new(
                            neighbor.x.clamp(0, CHUNK_AXIS as i32 - 1),
                            neighbor.y.clamp(0, CHUNK_AXIS as i32 - 1),
                            neighbor.z.clamp(0, CHUNK_AXIS as i32 - 1),
                        );

                        blocks.push((
                            (true_position + 1).as_uvec3(),
                            all[&position][bevy_world
                                .get::<Structure>(chunk_entity)
                                .unwrap()
                                .linearize(true_position.as_uvec3())],
                        ));

                        let inner_position = IVec3::new(
                            neighbor.x.clamp(1, CHUNK_AXIS as i32 - 2),
                            neighbor.y.clamp(1, CHUNK_AXIS as i32 - 2),
                            neighbor.z.clamp(1, CHUNK_AXIS as i32 - 2),
                        );

                        blocks.push((
                            (inner_position + 1).as_uvec3(),
                            all[&position][bevy_world
                                .get::<Structure>(chunk_entity)
                                .unwrap()
                                .linearize(inner_position.as_uvec3())],
                        ));
                    }
                }
            }
        }

        let mut greater_structure = Structure::new(UVec3::new(
            CHUNK_AXIS as u32 + 2,
            CHUNK_AXIS as u32 + 2,
            CHUNK_AXIS as u32 + 2,
        ));

        greater_structure.set_block(blocks);

        let index = 0..greater_structure.count() as u64;
        calc_ao(&mut greater_structure, index.clone());
        calc_cull(&mut greater_structure, index);

        let range = (0..greater_structure.count()).map(|i| greater_structure.delinearize(i));

        let gs_cull = greater_structure
            .get_cull(range.clone())
            .collect::<Vec<_>>();
        let gs_ao = greater_structure.get_ao(range).collect::<Vec<_>>();

        let mut chunk = bevy_world.get_mut::<Structure>(chunk_entity).unwrap();
        let mut cull = vec![];
        let mut ao = vec![];
        for d in 0..3 {
            for n in -1..=1 {
                let mut local = IVec3::ZERO;
                if n == -1 {
                    local[d] = 0;
                } else {
                    local[d] = CHUNK_AXIS as i32 - 1;
                }
                for u in 0..CHUNK_AXIS as i32 {
                    for v in 0..CHUNK_AXIS as i32 {
                        local[(d + 1) % 3] = u;
                        local[(d + 2) % 3] = v;
                        let gs_position = (local + 1).as_uvec3();

                        cull.push((
                            local.as_uvec3(),
                            gs_cull[greater_structure.linearize(gs_position)],
                        ));
                        ao.push((
                            local.as_uvec3(),
                            gs_ao[greater_structure.linearize(gs_position)],
                        ));
                    }
                }
            }
        }
        chunk.set_cull(cull);
        chunk.set_ao(ao);
        drop(chunk);
        bevy_world.entity_mut(chunk_entity).remove::<Dirty>();
    }
}

fn mesh(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &Structure, &Chunk), (Without<Active>, Without<Dirty>)>,
) {
    for (entity, structure, Chunk(position)) in query.iter() {
        dbg!("yo1212");
        let cube_mesh_handle: Handle<Mesh> = meshes.add(create_structure_mesh(&structure));
        let material = materials.add(StandardMaterial {
            base_color: Color::Rgba {
                red: 1.0,
                green: 1.0,
                blue: 1.0,
                alpha: 1.0,
            },
            metallic: 0.0,
            reflectance: 0.1,
            ..default()
        });
        commands.entity(entity).insert((
            Active,
            PbrBundle {
                mesh: cube_mesh_handle,
                material,
                transform: Transform {
                    translation: position.as_vec3() * CHUNK_AXIS as f32,
                    ..default()
                },
                ..default()
            },
        ));
    }
}

fn load(
    query1: Query<(&Camera, &Parent)>,
    query2: Query<(&Transform)>,
    mut world: ResMut<World>,
    mut commands: Commands,
) {
    let (_, camera_parent) = query1.single();
    let camera_transform = query2.get(camera_parent.get()).unwrap();
    let position = camera_transform.translation.as_ivec3() / CHUNK_AXIS as i32;
    if position == world.origin {
        return;
    }

    world.origin = position;

    let mut needed = HashSet::new();

    let view = world.view as i32;
    for x in -view..=view {
        for y in -1..=2 {
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
            AsyncComputeTaskPool::get_or_init(|| TaskPoolBuilder::new().num_threads(7).build())
                .spawn(async move {
                    let mut chunk = gen_chunk(position);
                    let index = 0..chunk.count() as u64;
                    calc_ao(&mut chunk, index.clone());
                    calc_cull(&mut chunk, index);
                    dbg!("yo");
                    (position, chunk)
                });

        world.chunk_futures.as_mut().unwrap().push(chunk_future);
        world.loaded.insert(position);
    }
}

fn gen_chunk(position: IVec3) -> Structure {
    let mut chunk = Structure::new(UVec3::new(
        CHUNK_AXIS as u32,
        CHUNK_AXIS as u32,
        CHUNK_AXIS as u32,
    ));
    let UVec3 {
        x: sx,
        y: sy,
        z: sz,
    } = chunk.size();
    let perlin = noise::Fbm::<noise::Perlin>::new(400);
    const NOISE_SCALE: i32 = 32;

    let mut noise_values = vec![];

    for z in 0..=CHUNK_AXIS as i32 / NOISE_SCALE {
        for y in 0..=CHUNK_AXIS as i32 / NOISE_SCALE {
            for x in 0..=CHUNK_AXIS as i32 / NOISE_SCALE {
                let nx = position.x * CHUNK_AXIS as i32 + x * NOISE_SCALE;
                let ny = position.y * CHUNK_AXIS as i32 + y * NOISE_SCALE;
                let nz = position.z * CHUNK_AXIS as i32 + z * NOISE_SCALE;
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

    let mut blocks = vec![];
    let smx = sx as usize / NOISE_SCALE as usize + 1;
    let smy = sy as usize / NOISE_SCALE as usize + 1;
    for z in 0..CHUNK_AXIS as u32 {
        for x in 0..CHUNK_AXIS as u32 {
            for y in 0..CHUNK_AXIS as u32 {
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

                let density = lerp3d(
                    x0y0z0,
                    x1y0z0,
                    x0y1z0,
                    x1y1z0,
                    x0y0z1,
                    x1y0z1,
                    x0y1z1,
                    x1y1z1,
                    ix as f64 / NOISE_SCALE as f64,
                    iy as f64 / NOISE_SCALE as f64,
                    iz as f64 / NOISE_SCALE as f64,
                );

                let density_mod = (32isize - ny as isize) as f64 * 0.035;
                blocks.push((
                    UVec3 { x, y, z },
                    if density + density_mod > 0.0 {
                        Block::Grass
                    } else {
                        Block::Air
                    },
                ));
            }
        }
    }
    chunk.set_block(blocks);
    chunk
}

fn calc_cull(structure: &mut Structure, index: impl Iterator<Item = u64>) {
    let UVec3 {
        x: sx,
        y: sy,
        z: sz,
    } = structure.size();

    let index = index.collect::<Vec<_>>();
    let range = (0..structure.count()).map(|i| structure.delinearize(i));
    let blocks = structure.get_block(range).collect::<Vec<_>>();
    let mut cull = structure
        .get_cull(
            index
                .iter()
                .copied()
                .map(|i| structure.delinearize(i as usize)),
        )
        .collect::<Vec<_>>()
        .into_iter()
        .zip(index)
        .map(|(mut direction, index)| {
            let position = structure.delinearize(index as usize);
            let mut dir_iter = (0..6)
                .map(|x| 1 << x)
                .map(Direction::from_bits)
                .map(Option::unwrap);
            for d in 0..2 {
                for n in (-1..=1).step_by(2) {
                    let current_direction = dir_iter.next().unwrap();
                    let mut normal = IVec3::default();
                    normal[d] = n;
                    let neighbor = (position.as_ivec3() + normal).as_uvec3();
                    if neighbor.x >= sx || neighbor.y >= sy || neighbor.z >= sz {
                        continue;
                    }
                    if neighbor.x < sx && neighbor.y < sy && neighbor.z < sz {
                        if matches!(blocks[structure.linearize(neighbor)], Block::Air) {
                            direction |= current_direction;
                        } else {
                            direction &= !current_direction;
                        }
                    }
                }
            }
            (position, direction)
        })
        .collect::<Vec<_>>();

    structure.set_cull(cull);
}

fn all_neighbors(position: IVec3, mut f: impl FnMut(IVec3)) {
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                (f)(position + IVec3 { x, y, z });
            }
        }
    }
}

fn all_neighbors_present(chunk_mappings: &HashMap<IVec3, Entity>, position: IVec3) -> bool {
    let mut neighbors = 0;
    all_neighbors(position, |neighbor| {
        if chunk_mappings.contains_key(&neighbor) {
            neighbors += 1;
        }
    });
    neighbors == 3usize.pow(3)
}

fn calc_ao(structure: &mut Structure, index: impl Iterator<Item = u64>) {
    let index = index.collect::<Vec<_>>();
    let range = (0..structure.count()).map(|i| structure.delinearize(i));
    let blocks = structure.get_block(range).collect::<Vec<_>>();
    let mut ao = structure
        .get_ao(
            index
                .iter()
                .copied()
                .map(|i| structure.delinearize(i as usize)),
        )
        .collect::<Vec<_>>()
        .into_iter()
        .zip(index)
        .map(|(mut ao, index)| {
            let position = structure.delinearize(index as usize);
            let mut dir_iter = (0..6)
                .map(|x| 1 << x)
                .map(Direction::from_bits)
                .map(Option::unwrap);
            for d in 0..3 {
                for n in (-1..=1).step_by(2) {
                    let current_direction = dir_iter.next().unwrap();
                    let mut normal = IVec3::default();
                    normal[d] = n;
                    let direction_index = current_direction.bits().trailing_zeros() as usize;
                    ao[direction_index] = voxel_ao(
                        structure,
                        &blocks,
                        position.as_ivec3() + normal,
                        IVec3 {
                            x: normal.z.abs(),
                            y: normal.x.abs(),
                            z: normal.y.abs(),
                        },
                        IVec3 {
                            x: normal.y.abs(),
                            y: normal.z.abs(),
                            z: normal.x.abs(),
                        },
                    );
                }
            }
            (position, ao)
        })
        .collect::<Vec<_>>();

    structure.set_ao(ao);
}

fn voxel_ao(structure: &Structure, blocks: &[Block], pos: IVec3, d1: IVec3, d2: IVec3) -> Vec4 {
    let UVec3 {
        x: sx,
        y: sy,
        z: sz,
    } = structure.size();
    let voxel_present = |pos: IVec3| -> f32 {
        let pos = pos.as_uvec3();
        if pos.x >= sx || pos.y >= sy || pos.z >= sz {
            0.0
        } else {
            !matches!(blocks[structure.linearize(pos)], Block::Air) as i32 as f32
        }
    };
    let vertex_ao =
        |side: Vec2, corner: f32| (side.x + side.y + f32::max(corner, side.x * side.y)) / 3.0;
    let side = Vec4::new(
        (voxel_present)(pos + d1),
        (voxel_present)(pos + d2),
        (voxel_present)(pos - d1),
        (voxel_present)(pos - d2),
    );
    let corner = Vec4::new(
        (voxel_present)(pos + d1 + d2),
        (voxel_present)(pos - d1 + d2),
        (voxel_present)(pos - d1 - d2),
        (voxel_present)(pos + d1 - d2),
    );
    1.0 - Vec4::new(
        (vertex_ao)(Vec2::new(side.x, side.y), corner.x),
        (vertex_ao)(Vec2::new(side.y, side.z), corner.y),
        (vertex_ao)(Vec2::new(side.z, side.w), corner.z),
        (vertex_ao)(Vec2::new(side.w, side.x), corner.w),
    )
}

pub trait RayExt {
    fn intersect_voxels(self, bevy_world: &bevy::prelude::World) -> Option<f32>;
}

pub struct VoxelRay {
    position: IVec3,
    mask: BVec3,
    fmask: Vec3,
    imask: IVec3,
    side_dist: Vec3,
    delta_dist: Vec3,
    ray_step: IVec3,
    distance: f32,
    step_count: usize,
}

fn voxel_ray(ray: Ray) -> VoxelRay {
    let mut position = ray.origin.floor().as_ivec3();
    let mut mask = BVec3::splat(false);
    let mut side_dist = ray.direction.signum()
        * ((ray.origin.floor() - ray.origin) + (ray.direction.signum() * 0.5) + 0.5);
    let delta_dist = 1.0 / ray.direction.abs();
    let ray_step = ray.direction.signum().as_ivec3();
    let fmask = default();
    let imask = default();
    let distance = default();
    let step_count = default();
    VoxelRay {
        position,
        mask,
        side_dist,
        delta_dist,
        ray_step,
        fmask,
        imask,
        distance,
        step_count,
    }
}

fn voxel_step(ray: &mut VoxelRay) {
    let VoxelRay {
        position,
        mask,
        side_dist,
        delta_dist,
        ray_step,
        fmask,
        imask,
        distance,
        step_count,
    } = ray;
    mask.x = side_dist.x <= side_dist.y.min(side_dist.z);
    mask.y = side_dist.y <= side_dist.z.min(side_dist.x);
    mask.z = side_dist.z <= side_dist.x.min(side_dist.y);

    *imask = IVec3::new(mask.x as i32, mask.y as i32, mask.z as i32);
    *fmask = Vec3::new(imask.x as f32, imask.y as f32, imask.z as f32);

    *side_dist += *fmask * *delta_dist;
    *position += *imask * *ray_step;
    *distance = (*fmask * (*side_dist - *delta_dist)).length();
    *step_count += 1;
}

impl RayExt for bevy::math::Ray {
    fn intersect_voxels(mut self, bevy_world: &bevy::prelude::World) -> Option<f32> {
        self.direction = self.direction.normalize();
        let mut ray = voxel_ray(self);

        loop {
            if ray.step_count >= 8192 {
                return None;
            }

            let chunk_position = ray.position.div_euclid(IVec3::splat(CHUNK_AXIS as i32));

            if let Some(&chunk_entity) = bevy_world.resource::<World>().mapping.get(&chunk_position)
            {
                let chunk = bevy_world.get::<Structure>(chunk_entity).unwrap();

                let local_position = ray
                    .position
                    .rem_euclid(IVec3::splat(CHUNK_AXIS as i32))
                    .as_uvec3();

                if !matches!(
                    chunk.get_block(iter::once(local_position)).next().unwrap(),
                    Block::Air
                ) {
                    return Some(ray.distance);
                }
            }
            voxel_step(&mut ray);
        }
    }
}

fn cast_system(bevy_world: &mut bevy::prelude::World) {
    let mouse = bevy_world.resource::<Input<MouseButton>>();
    if mouse.just_pressed(MouseButton::Left) {
        let viewport_position = {
            let mut system_state = SystemState::<Query<(&Window)>>::new(bevy_world);
            let query = system_state.get(bevy_world);
            let window = query.single();
            window.cursor_position().unwrap()
        };
        let ray = {
            let mut system_state =
                SystemState::<Query<(&GlobalTransform, &Camera)>>::new(bevy_world);
            let query = system_state.get(bevy_world);
            let (transform, camera) = query.single();
            camera
                .viewport_to_world(transform, viewport_position)
                .unwrap()
        };
        let distance = ray.intersect_voxels(bevy_world).unwrap();
        let position = (ray.origin + ray.direction * distance).as_ivec3();
        bevy_world.resource_mut::<BuildTool>().points.push(position);
    }
}

fn set_block(bevy_world: &mut bevy::prelude::World, position: IVec3, block: Block) {
    let chunk_position = position.div_euclid(IVec3::splat(CHUNK_AXIS as i32));

    if let Some(&chunk_entity) = bevy_world.resource::<World>().mapping.get(&chunk_position) {
        let mut chunk = bevy_world.get_mut::<Structure>(chunk_entity).unwrap();

        let local_position = position
            .rem_euclid(IVec3::splat(CHUNK_AXIS as i32))
            .as_uvec3();

        chunk.set_block(iter::once((local_position, block)));

        bevy_world.entity_mut(chunk_entity).remove::<Active>();
        bevy_world.entity_mut(chunk_entity).insert(Dirty);
    }
}

fn get_block(bevy_world: &mut bevy::prelude::World, position: IVec3) -> Option<Block> {
    let chunk_position = position.div_euclid(IVec3::splat(CHUNK_AXIS as i32));

    if let Some(&chunk_entity) = bevy_world.resource::<World>().mapping.get(&chunk_position) {
        let mut chunk = bevy_world.get_mut::<Structure>(chunk_entity).unwrap();

        let local_position = position
            .rem_euclid(IVec3::splat(CHUNK_AXIS as i32))
            .as_uvec3();

        return chunk.get_block(iter::once(local_position)).next();
    }
    None
}

fn get_ground_level(bevy_world: &mut bevy::prelude::World, mut position: IVec3) -> i32 {
    while !matches!(get_block(bevy_world, position), Some(Block::Air)) {
        position += 1;
    }
    position.y
}

bitflags! {
    #[derive(PartialEq, Eq, Clone, Copy)]
    pub struct LineMode: u64 {
        const NONE =    0b00000000;
        const MAJOR =    0b00000001;
        const MINOR =   0b00000010;
        const BOTH =    0b00000011;
    }
}

fn draw_line(start: IVec3, end: IVec3, mode: LineMode, mut fill: impl FnMut(IVec3)) {
    let mut delta = IVec3::ZERO;
    let mut delta_x2 = IVec3::ZERO;
    let mut err = IVec3::ZERO;
    let mut step = IVec3::ZERO;

    delta = end - start;

    for d in 0..3 {
        if delta[d] < 0 {
            delta[d] *= -1;
            step[d] = -1;
        } else {
            step[d] = 1;
        }
    
        delta_x2[d] = delta[d] * 2;
    }

    let mut pos = start;

    (fill)(pos);

    let u;
    if delta.x >= delta.y && delta.x >= delta.z {
        u = 0;
    } else if delta.y >= delta.x && delta.y >= delta.z {
        u = 1;
    } else {
        u = 2;
    }

    let v = (u + 1) % 3;
    let w = (u + 2) % 3;

    err[u] = delta_x2[v] - delta[u];
    err[v] = delta_x2[w] - delta[u];

    while pos[u] != end[u] {
        pos[u] += step[u];
       
        if err[u] >= 0 {
            if mode & LineMode::MAJOR != LineMode::empty() {
                (fill)(pos);
            }
            pos[v] += step[v];
            if mode & LineMode::MINOR != LineMode::empty() {
                let mut back = pos;
                back[u] -= step[u];
                (fill)(back);
            }
            err[u] -= delta_x2[u];
        }
        if err[v] >= 0 {
            if mode & LineMode::MAJOR != LineMode::empty() {
                (fill)(pos);
            }
            pos[w] += step[w];
            if mode & LineMode::MINOR != LineMode::empty() {
                let mut back = pos;
                back[u] -= step[u];
                (fill)(back);
            }
            err[v] -= delta_x2[u];
        }
        err[u] += delta_x2[v];
        err[v] += delta_x2[w];
        (fill)(pos);
    }
}

fn build_road(bevy_world: &mut bevy::prelude::World) {
    let points = &mut bevy_world.resource_mut::<BuildTool>().points;
    if points.len() < 4 {
        return;
    }
    let d = points.pop().unwrap();
    let c = points.pop().unwrap();
    let b = points.pop().unwrap();
    let a = points.pop().unwrap();
    *points = vec![];
    use bezier_nd::*;
    use geo_nd::*;
    let a = FArray::from(a.as_vec3().to_array());
    let b = FArray::from(b.as_vec3().to_array());
    let c = FArray::from(c.as_vec3().to_array());
    let d = FArray::from(d.as_vec3().to_array());
    
    let curve = Bezier::cubic(&a, &b, &c, &d);

    for (a, b) in curve.as_lines(0.01) {
        let mut a = Vec3::from_array(a.into()).as_ivec3();
        let mut b = Vec3::from_array(b.into()).as_ivec3();
        a.y = get_ground_level(bevy_world, a);
        b.y = get_ground_level(bevy_world, b);
        draw_line(a, b, LineMode::MAJOR, |pos| set_block(bevy_world, pos, Block::Stone));
    }
}

#[derive(Resource, Default)]
pub struct BuildTool {
    points: Vec<IVec3>,
}

fn main() {
    let mut app = App::new();

    app.insert_resource(World {
        view: 5,
        origin: IVec3 {
            x: i32::MAX,
            y: 0,
            z: 0,
        },
        loaded: HashSet::new(),
        mapping: HashMap::new(),
        chunk_futures: Some(Vec::new()),
    });
    app.insert_resource(DirectionalLightShadowMap { size: 4096 });
    app.init_resource::<BuildTool>();
    app.add_plugins(DefaultPlugins);
    app.add_systems(Startup, setup)
        .add_systems(Update, load)
        .add_systems(Update, spawn)
        .add_systems(Update, mesh)
        .add_systems(Update, camera)
        .add_systems(Update, cast_system)
        .add_systems(Update, build_road)
        .add_systems(Update, (spawn, apply_deferred, consolidate).chain());

    app.run();
}
